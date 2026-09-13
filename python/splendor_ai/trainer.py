"""Trainer implementation with multi-task loss, AMP and atomic checkpointing."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from splendor_ai.dataset import SplendorDataset
from splendor_ai.net import SplendorNet


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    value_loss_coeff: float = 1.0
    grad_clip_norm: float = 5.0
    batch_size: int = 128
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

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练单个 Epoch."""
        self.net.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0

        for batch in dataloader:
            obs = batch["obs"].to(self.device)
            mask = batch["mask"].to(self.device)
            target_policy = batch["target_policy"].to(self.device)
            target_value = batch["target_value"].to(self.device)
            b_size = obs.shape[0]

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits, value = self.net(obs)
                masked_logits = SplendorNet.mask_logits(logits, mask)

                # 1. 策略交叉熵损失
                log_probs = F.log_softmax(masked_logits, dim=-1)
                # target_policy 可能是 One-Hot 或连续概率分布
                policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

                # 2. 价值 MSE 损失
                value_loss = F.mse_loss(value, target_value)

                # 联合损失
                loss = policy_loss + self.cfg.value_loss_coeff * value_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * b_size
            total_p_loss += policy_loss.item() * b_size
            total_v_loss += value_loss.item() * b_size

            # 评估命中率
            pred_top3 = masked_logits.topk(k=min(3, masked_logits.shape[-1]), dim=-1).indices
            target_idx = target_policy.argmax(dim=-1, keepdim=True)
            correct_top1 += (pred_top3[:, :1] == target_idx).sum().item()
            correct_top3 += (pred_top3 == target_idx).any(dim=-1).sum().item()
            total_samples += b_size

        self.epoch += 1
        self.scheduler.step()

        return {
            "loss": total_loss / total_samples,
            "policy_loss": total_p_loss / total_samples,
            "value_loss": total_v_loss / total_samples,
            "top1_acc": correct_top1 / total_samples,
            "top3_acc": correct_top3 / total_samples,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估当前模型在验证集上的表现."""
        self.net.eval()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                obs = batch["obs"].to(self.device)
                mask = batch["mask"].to(self.device)
                target_policy = batch["target_policy"].to(self.device)
                target_value = batch["target_value"].to(self.device)
                b_size = obs.shape[0]

                logits, value = self.net(obs)
                masked_logits = SplendorNet.mask_logits(logits, mask)

                log_probs = F.log_softmax(masked_logits, dim=-1)
                policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(value, target_value)
                loss = policy_loss + self.cfg.value_loss_coeff * value_loss

                total_loss += loss.item() * b_size
                total_p_loss += policy_loss.item() * b_size
                total_v_loss += value_loss.item() * b_size

                pred_top3 = masked_logits.topk(k=min(3, masked_logits.shape[-1]), dim=-1).indices
                target_idx = target_policy.argmax(dim=-1, keepdim=True)
                correct_top1 += (pred_top3[:, :1] == target_idx).sum().item()
                correct_top3 += (pred_top3 == target_idx).any(dim=-1).sum().item()
                total_samples += b_size

        return {
            "eval_loss": total_loss / total_samples,
            "eval_policy_loss": total_p_loss / total_samples,
            "eval_value_loss": total_v_loss / total_samples,
            "eval_top1_acc": correct_top1 / total_samples,
            "eval_top3_acc": correct_top3 / total_samples,
        }

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

        # 写入临时文件后执行原子替换
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{filename}.", suffix=".tmp", dir=self.ckpt_dir, delete=False
        ) as f:
            tmp_path = Path(f.name)

        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, ckpt_path)
            if is_best:
                best_path = self.ckpt_dir / "best.pt"
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
