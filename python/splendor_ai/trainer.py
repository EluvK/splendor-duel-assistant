"""Trainer implementation with multi-task loss, AMP, atomic checkpointing and sharded streaming."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from splendor_ai.dataset import CompactBatch, FastTensorLoader
from splendor_ai.net import SplendorNet
from splendor_ai.progress import Progress


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    value_loss_coeff: float = 1.0
    win_loss_coeff: float = 1.0
    turns_loss_coeff: float = 0.5
    reason_loss_coeff: float = 0.3
    grad_clip_norm: float = 5.0
    batch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    ckpt_dir: str = "checkpoints"
    t_max_epochs: int = 10


class Trainer:
    """策略价值网络综合训练器."""

    def __init__(self, net: SplendorNet, config: Optional[TrainerConfig] = None) -> None:
        self.cfg = config or TrainerConfig()
        self.device = torch.device(self.cfg.device)
        self.net = net.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.t_max_epochs,
            eta_min=self.cfg.min_lr,
        )

        use_amp = self.cfg.amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.amp_enabled = use_amp

        self.epoch = 0
        self.ckpt_dir = Path(self.cfg.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: Any) -> Dict[str, float]:
        """训练单个 Epoch (单一 DataLoader/FastTensorLoader)."""
        self.net.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_win_loss = 0.0
        total_turns_loss = 0.0
        total_reason_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0

        pbar = Progress(total=len(dataloader), label=f"Train Ep {self.epoch + 1}")

        for step_i, batch in enumerate(dataloader):
            obs = batch["obs"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            target_action = batch["action"].to(self.device, non_blocking=True)
            target_value = batch["value"].to(self.device, non_blocking=True)
            target_reason = batch.get("reason")
            if target_reason is not None:
                target_reason = target_reason.to(self.device, non_blocking=True)
            else:
                target_reason = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)

            target_win = target_value[:, 0:1]
            target_turns = target_value[:, 1:2]
            b_size = obs.shape[0]

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits, win_v, turns_v, reason_logits = self.net(obs)
                masked_logits = SplendorNet.mask_logits(logits, mask)

                policy_loss = F.cross_entropy(masked_logits, target_action)
                win_loss = F.mse_loss(win_v, target_win)
                turns_loss = F.smooth_l1_loss(turns_v, target_turns)
                reason_loss = F.cross_entropy(reason_logits, target_reason)

                value_loss = (
                    self.cfg.win_loss_coeff * win_loss
                    + self.cfg.turns_loss_coeff * turns_loss
                    + self.cfg.reason_loss_coeff * reason_loss
                )
                loss = policy_loss + self.cfg.value_loss_coeff * value_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * b_size
            total_p_loss += policy_loss.item() * b_size
            total_v_loss += value_loss.item() * b_size
            total_win_loss += win_loss.item() * b_size
            total_turns_loss += turns_loss.item() * b_size
            total_reason_loss += reason_loss.item() * b_size

            pred_top3 = masked_logits.topk(k=3, dim=-1).indices
            correct_top1 += (pred_top3[:, 0] == target_action).sum().item()
            correct_top3 += (pred_top3 == target_action.unsqueeze(1)).any(dim=-1).sum().item()
            total_samples += b_size

            if (step_i + 1) % 10 == 0 or (step_i + 1) == len(dataloader):
                pbar.update(
                    step_i + 1,
                    extra=f"loss: {loss.item():.3f} | win: {win_loss.item():.3f} | top1: {correct_top1/total_samples*100:.1f}%",
                )

        pbar.done(f"loss: {total_loss/total_samples:.4f} | top1: {correct_top1/total_samples*100:.1f}%")
        self.epoch += 1
        self.scheduler.step()

        return {
            "loss": total_loss / total_samples,
            "policy_loss": total_p_loss / total_samples,
            "value_loss": total_v_loss / total_samples,
            "win_loss": total_win_loss / total_samples,
            "turns_loss": total_turns_loss / total_samples,
            "reason_loss": total_reason_loss / total_samples,
            "top1_acc": correct_top1 / total_samples,
            "top3_acc": correct_top3 / total_samples,
            "lr": self.optimizer.param_groups[0]["lr"],
            "total_samples": total_samples,
        }

    def train_epoch_sharded(
        self,
        shard_files: List[Path],
        batch_size: int = 256,
        shuffle_shards: bool = True,
    ) -> Dict[str, float]:
        """在多个分片文件流上训练单个完整 Epoch (内存永远恒定在单分片上限内)."""
        self.net.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0

        files = list(shard_files)
        if shuffle_shards:
            random.shuffle(files)

        num_shards = len(files)
        pbar = Progress(total=num_shards, label=f"Train Ep {self.epoch + 1} Shards")

        def _load_loader(shard_p: Path) -> FastTensorLoader:
            batch_d = CompactBatch.load_npz(shard_p)
            return FastTensorLoader(batch_d, batch_size=batch_size, shuffle=True, device=self.device)

        # 异步预加载流水线 (双缓冲)：后台线程在 GPU 计算当前分片时提前加载下一分片，消除 GPU 等待横跳
        with ThreadPoolExecutor(max_workers=1) as prefetcher:
            next_future = prefetcher.submit(_load_loader, files[0]) if num_shards > 0 else None

            for s_idx, shard_path in enumerate(files):
                # 瞬间获取已就绪的分片加载器
                loader = next_future.result()

                # 立即向后台调度下一个分片的 I/O 与反序列化
                if s_idx + 1 < num_shards:
                    next_future = prefetcher.submit(_load_loader, files[s_idx + 1])
                else:
                    next_future = None

                n_batches = len(loader)
                for b_idx, batch in enumerate(loader):
                    obs = batch["obs"]
                    mask = batch["mask"]
                    target_action = batch["action"]
                    target_value = batch["value"]
                    target_win = target_value[:, 0:1]
                    target_turns = target_value[:, 1:2]
                    target_reason = batch.get("reason")
                    if target_reason is not None:
                        target_reason = target_reason.to(self.device, non_blocking=True)
                    else:
                        target_reason = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)

                    b_size = obs.shape[0]

                    self.optimizer.zero_grad()

                    with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                        logits, win_v, turns_v, reason_logits = self.net(obs)
                        masked_logits = SplendorNet.mask_logits(logits, mask)

                        policy_loss = F.cross_entropy(masked_logits, target_action)
                        win_loss = F.mse_loss(win_v, target_win)
                        turns_loss = F.smooth_l1_loss(turns_v, target_turns)
                        reason_loss = F.cross_entropy(reason_logits, target_reason)

                        value_loss = (
                            self.cfg.win_loss_coeff * win_loss
                            + self.cfg.turns_loss_coeff * turns_loss
                            + self.cfg.reason_loss_coeff * reason_loss
                        )
                        loss = policy_loss + self.cfg.value_loss_coeff * value_loss

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    total_loss += loss.item() * b_size
                    total_p_loss += policy_loss.item() * b_size
                    total_v_loss += value_loss.item() * b_size

                    pred_top3 = masked_logits.topk(k=3, dim=-1).indices
                    correct_top1 += (pred_top3[:, 0] == target_action).sum().item()
                    correct_top3 += (pred_top3 == target_action.unsqueeze(1)).any(dim=-1).sum().item()
                    total_samples += b_size

                    # 分片内平滑进度推进 (内置 0.15s 节流，零性能损耗)
                    frac_done = s_idx + (b_idx + 1) / max(n_batches, 1)
                    cur_top1 = (correct_top1 / max(total_samples, 1)) * 100.0
                    cur_loss = total_loss / max(total_samples, 1)
                    pbar.update(frac_done, extra=f"loss: {cur_loss:.3f} | top1: {cur_top1:.1f}%")

                del loader

        pbar.done(f"loss: {total_loss/total_samples:.4f} | top1: {correct_top1/total_samples*100:.1f}%")
        self.epoch += 1
        self.scheduler.step()

        return {
            "loss": total_loss / total_samples,
            "policy_loss": total_p_loss / total_samples,
            "value_loss": total_v_loss / total_samples,
            "top1_acc": correct_top1 / total_samples,
            "top3_acc": correct_top3 / total_samples,
            "lr": self.optimizer.param_groups[0]["lr"],
            "total_samples": total_samples,
        }

    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        """评估当前模型在验证集上的表现."""
        self.net.eval()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0

        pbar = Progress(total=len(dataloader), label="Evaluate")

        with torch.no_grad():
            for step_i, batch in enumerate(dataloader):
                obs = batch["obs"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                target_action = batch["action"].to(self.device, non_blocking=True)
                target_value = batch["value"].to(self.device, non_blocking=True)
                target_reason = batch.get("reason")
                if target_reason is not None:
                    target_reason = target_reason.to(self.device, non_blocking=True)
                else:
                    target_reason = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)

                target_win = target_value[:, 0:1]
                target_turns = target_value[:, 1:2]
                b_size = obs.shape[0]

                logits, win_v, turns_v, reason_logits = self.net(obs)
                masked_logits = SplendorNet.mask_logits(logits, mask)

                policy_loss = F.cross_entropy(masked_logits, target_action)
                win_loss = F.mse_loss(win_v, target_win)
                turns_loss = F.smooth_l1_loss(turns_v, target_turns)
                reason_loss = F.cross_entropy(reason_logits, target_reason)

                value_loss = (
                    self.cfg.win_loss_coeff * win_loss
                    + self.cfg.turns_loss_coeff * turns_loss
                    + self.cfg.reason_loss_coeff * reason_loss
                )
                loss = policy_loss + self.cfg.value_loss_coeff * value_loss

                total_loss += loss.item() * b_size
                total_p_loss += policy_loss.item() * b_size
                total_v_loss += value_loss.item() * b_size

                pred_top3 = masked_logits.topk(k=3, dim=-1).indices
                correct_top1 += (pred_top3[:, 0] == target_action).sum().item()
                correct_top3 += (pred_top3 == target_action.unsqueeze(1)).any(dim=-1).sum().item()
                total_samples += b_size

                if (step_i + 1) % 10 == 0 or (step_i + 1) == len(dataloader):
                    pbar.update(step_i + 1)

        pbar.done()

        return {
            "eval_loss": total_loss / total_samples,
            "eval_policy_loss": total_p_loss / total_samples,
            "eval_value_loss": total_v_loss / total_samples,
            "eval_top1_acc": correct_top1 / total_samples,
            "eval_top3_acc": correct_top3 / total_samples,
        }

    def evaluate_sharded(self, val_shard_path: Path, batch_size: int = 256) -> Dict[str, float]:
        """评估单个分片文件."""
        val_data = CompactBatch.load_npz(val_shard_path)
        loader = FastTensorLoader(val_data, batch_size=batch_size, shuffle=False, device=self.device)
        res = self.evaluate(loader)
        del val_data
        del loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    def save_checkpoint(
        self, filename: str = "latest.pt", is_best: bool = False, meta: Optional[Dict[str, Any]] = None
    ) -> Path:
        """原子级安全保存检查点."""
        ckpt_path = self.ckpt_dir / filename
        payload = {
            "epoch": self.epoch,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.amp_enabled else None,
            "meta": meta or {},
        }

        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{filename}.", suffix=".tmp", dir=self.ckpt_dir, delete=False
        ) as f:
            tmp_path = Path(f.name)

        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, ckpt_path)
            if is_best:
                best_path = self.ckpt_dir / "best.pt"
                if ckpt_path.resolve() != best_path.resolve():
                    import shutil
                    shutil.copyfile(ckpt_path, best_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return ckpt_path

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """加载检查点权重与训练状态."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] and self.amp_enabled:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.epoch = checkpoint.get("epoch", 0)
        return checkpoint.get("meta", {})
