"""Compact, vectorized, and sharded dataset implementations for Splendor Duel."""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from splendor_ai.env import SplendorDuelEnv


class CompactBatch:
    """紧凑内存样本块 (零多余对象开销，纯连续扁平数组)."""

    obs: np.ndarray  # [N, OBS_SIZE] (969) float32
    mask: np.ndarray  # [N, ACTION_SIZE] (1856) bool
    target_policy: np.ndarray  # [N, ACTION_SIZE] (1856) float32 (MCTS visits 软概率分布)
    value: np.ndarray  # [N, 2] float32 (col 0: 纯胜负期望, col 1: 归一化剩余轮数)
    reason: np.ndarray  # [N, 6] float32 区分归属的多标签独立胜因 [我方3项, 敌方3项]

    def __init__(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        value: np.ndarray,
        reason: np.ndarray,
        target_policy: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
    ) -> None:
        self.obs = obs
        self.mask = mask
        self.value = value
        n = len(obs)

        # 适配 reason 形状 (支持 6 维区分胜负归属多标签，兼容老版本 3 维或 1D 标量)
        if reason.ndim == 1:
            r_2d = np.zeros((n, 6), dtype=np.float32)
            for i, r in enumerate(reason):
                if 0 <= r < 6:
                    r_2d[i, r] = 1.0
            self.reason = r_2d
        elif reason.shape[1] == 3:
            pad = np.zeros((n, 3), dtype=np.float32)
            self.reason = np.concatenate([reason, pad], axis=1).astype(np.float32)
        elif reason.shape[1] == 4:
            pad = np.zeros((n, 3), dtype=np.float32)
            self.reason = np.concatenate([reason[:, :3], pad], axis=1).astype(np.float32)
        else:
            self.reason = reason.astype(np.float32)

        if target_policy is not None:
            self.target_policy = target_policy
            self._action = action
        elif action is not None:
            # 由 action 自动构造 one-hot 策略分布以保持兼容
            tp = np.zeros((n, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32)
            for i, a in enumerate(action):
                if 0 <= a < SplendorDuelEnv.ACTION_SIZE:
                    tp[i, a] = 1.0
            self.target_policy = tp
            self._action = action
        else:
            self.target_policy = np.zeros((n, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32)
            self._action = None

    @property
    def action(self) -> np.ndarray:
        """保持向后兼容的标量动作索引 (从软概率分布中提取最大概率动作)."""
        if self._action is not None:
            return self._action
        if self.target_policy.ndim == 2 and len(self.target_policy) > 0:
            return self.target_policy.argmax(axis=-1).astype(np.int64)
        return np.zeros(len(self.obs), dtype=np.int64)

    @property
    def num_samples(self) -> int:
        return len(self.obs)

    def save_npz(self, path: Path, compressed: bool = True) -> None:
        """持久化保存为分片文件 (默认启用压缩，极大降低稀疏动作空间下的磁盘占用)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_fn = np.savez_compressed if compressed else np.savez
        save_fn(
            path,
            obs=self.obs,
            mask=self.mask,
            target_policy=self.target_policy,
            action=self.action,
            value=self.value,
            reason=self.reason,
        )

    @classmethod
    def load_npz(cls, path: Path) -> "CompactBatch":
        """从分片文件加载样本 (平滑兼容老旧单动作与单标签分片)."""
        data = np.load(path)
        obs = data["obs"]
        mask = data["mask"]
        n = len(obs)

        if "target_policy" in data:
            target_policy = data["target_policy"]
        elif "action" in data:
            actions = data["action"]
            target_policy = np.zeros((n, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32)
            for i, a in enumerate(actions):
                if 0 <= a < SplendorDuelEnv.ACTION_SIZE:
                    target_policy[i, a] = 1.0
        else:
            target_policy = np.zeros((n, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32)

        raw_val = data["value"]
        if raw_val.ndim == 1:
            raw_val = raw_val.reshape(-1, 1)
        if raw_val.shape[1] == 1:
            turns_pad = np.full_like(raw_val, 0.5, dtype=np.float32)
            val_2d = np.concatenate([raw_val, turns_pad], axis=1)
        else:
            val_2d = raw_val

        if "reason" in data:
            raw_reason = data["reason"]
            if raw_reason.ndim == 1:
                reason = np.zeros((n, 6), dtype=np.float32)
                for i, r in enumerate(raw_reason):
                    if 0 <= r < 6:
                        reason[i, r] = 1.0
            elif raw_reason.shape[1] == 3:
                pad = np.zeros((n, 3), dtype=np.float32)
                reason = np.concatenate([raw_reason, pad], axis=1).astype(np.float32)
            elif raw_reason.shape[1] == 4:
                pad = np.zeros((n, 3), dtype=np.float32)
                reason = np.concatenate([raw_reason[:, :3], pad], axis=1).astype(np.float32)
            else:
                reason = raw_reason.astype(np.float32)
        else:
            reason = np.zeros((n, 6), dtype=np.float32)

        act = data["action"] if "action" in data else None

        return cls(
            obs=obs,
            mask=mask,
            target_policy=target_policy,
            value=val_2d,
            reason=reason,
            action=act,
        )


class FastTensorLoader:
    """零单样本切片开销的整块张量向量化批处理加载器 (吞吐提升数倍)."""

    def __init__(
        self,
        batch: CompactBatch,
        batch_size: int = 256,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = batch.num_samples
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.device = device

        # 转为 PyTorch 张量
        self.obs = torch.from_numpy(batch.obs).float()
        self.mask = torch.from_numpy(batch.mask).bool()
        self.target_policy = torch.from_numpy(batch.target_policy).float()
        self.action = torch.from_numpy(batch.action).long()
        self.value = torch.from_numpy(batch.value).float()
        self.reason = torch.from_numpy(batch.reason).float()

        self.resident_on_device = False
        if device is not None and device.type == "cuda":
            try:
                # 单个分片通常约几百MB，直接常驻显存，零总线传输延迟
                self.obs = self.obs.to(device)
                self.mask = self.mask.to(device)
                self.target_policy = self.target_policy.to(device)
                self.action = self.action.to(device)
                self.value = self.value.to(device)
                self.reason = self.reason.to(device)
                self.resident_on_device = True
            except RuntimeError:
                # 显存不足时自动回退为 CPU 内存驻留
                self.resident_on_device = False

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            if self.resident_on_device:
                perm = torch.randperm(self.num_samples, device=self.device)
            else:
                perm = torch.randperm(self.num_samples)
        else:
            perm = None

        for b in range(self.num_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            if perm is not None:
                idx = perm[start:end]
            else:
                idx = slice(start, end)

            b_obs = self.obs[idx]
            b_mask = self.mask[idx]
            b_policy = self.target_policy[idx]
            b_action = self.action[idx]
            b_value = self.value[idx]
            b_reason = self.reason[idx]

            if not self.resident_on_device and self.device is not None:
                b_obs = b_obs.to(self.device, non_blocking=True)
                b_mask = b_mask.to(self.device, non_blocking=True)
                b_policy = b_policy.to(self.device, non_blocking=True)
                b_action = b_action.to(self.device, non_blocking=True)
                b_value = b_value.to(self.device, non_blocking=True)
                b_reason = b_reason.to(self.device, non_blocking=True)

            yield {
                "obs": b_obs,
                "mask": b_mask,
                "target_policy": b_policy,
                "action": b_action,
                "value": b_value,
                "reason": b_reason,
            }


class CompactDataset(Dataset):
    """基于紧凑连续内存数组的标准 PyTorch 数据集 (兼容传统 DataLoader)."""

    def __init__(self, batch: CompactBatch) -> None:
        self.obs = torch.from_numpy(batch.obs).float()
        self.mask = torch.from_numpy(batch.mask).bool()
        self.target_policy = torch.from_numpy(batch.target_policy).float()
        self.action = torch.from_numpy(batch.action).long()
        self.value = torch.from_numpy(batch.value).float()
        self.reason = torch.from_numpy(batch.reason).float()

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.obs[idx],
            "mask": self.mask[idx],
            "target_policy": self.target_policy[idx],
            "action": self.action[idx],
            "value": self.value[idx],
            "reason": self.reason[idx],
        }


class ShardedBuffer:
    """面向百万局级别训练的磁盘分片缓存管理器."""

    def __init__(self, shard_dir: Path) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.refresh()

    def refresh(self) -> None:
        self.shard_files: List[Path] = sorted(list(self.shard_dir.glob("shard_*.npz")))

    def add_shard(self, batch: CompactBatch, compressed: bool = True) -> Path:
        """保存新分片到磁盘."""
        shard_idx = len(self.shard_files) + 1
        path = self.shard_dir / f"shard_{shard_idx:05d}.npz"
        batch.save_npz(path, compressed=compressed)
        self.shard_files.append(path)
        return path

    def iter_shards(self, shuffle: bool = False) -> Iterator[CompactBatch]:
        """流式遍历分片，加载完一个训练完即释放，内存恒定稳定."""
        files = list(self.shard_files)
        if shuffle:
            random.shuffle(files)
        for p in files:
            yield CompactBatch.load_npz(p)

    def count_total_samples(self) -> int:
        """快速统计所有分片的总样本数."""
        total = 0
        for p in self.shard_files:
            with np.load(p) as data:
                if "target_policy" in data:
                    total += len(data["target_policy"])
                elif "action" in data:
                    total += len(data["action"])
                else:
                    total += len(data["obs"])
        return total

    def clear(self) -> None:
        """清空所有分片."""
        for p in self.shard_files:
            if p.exists():
                p.unlink()
        self.shard_files.clear()


class ReplayBuffer:
    """AlphaZero 自博弈经验回放池 (滑动窗口管理最近的高质量对局轨迹)."""

    def __init__(self, max_samples: int = 50000) -> None:
        self.max_samples = max_samples
        self.obs_list: List[np.ndarray] = []
        self.mask_list: List[np.ndarray] = []
        self.policy_list: List[np.ndarray] = []
        self.value_list: List[np.ndarray] = []
        self.reason_list: List[np.ndarray] = []
        self.total_samples = 0

    def add_batch(self, batch: CompactBatch) -> None:
        """存入一批新的自博弈数据，若超出容量则滑动淘汰最老的数据."""
        if batch.num_samples == 0:
            return
        self.obs_list.append(batch.obs)
        self.mask_list.append(batch.mask)
        self.policy_list.append(batch.target_policy)
        self.value_list.append(batch.value)
        self.reason_list.append(batch.reason)
        self.total_samples += batch.num_samples

        # 滑动窗口淘汰最老的一批
        while len(self.policy_list) > 1 and self.total_samples > self.max_samples:
            removed_count = len(self.policy_list[0])
            self.obs_list.pop(0)
            self.mask_list.pop(0)
            self.policy_list.pop(0)
            self.value_list.pop(0)
            self.reason_list.pop(0)
            self.total_samples -= removed_count

    def get_compact_batch(self) -> CompactBatch:
        """汇聚当前 Buffer 内全部有效样本为一个连续的 CompactBatch."""
        if not self.policy_list:
            return CompactBatch(
                obs=np.zeros((0, SplendorDuelEnv.OBS_SIZE), dtype=np.float32),
                mask=np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=bool),
                target_policy=np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32),
                value=np.zeros((0, 2), dtype=np.float32),
                reason=np.zeros((0, 6), dtype=np.float32),
            )
        if len(self.policy_list) == 1:
            return CompactBatch(
                obs=self.obs_list[0],
                mask=self.mask_list[0],
                target_policy=self.policy_list[0],
                value=self.value_list[0],
                reason=self.reason_list[0],
            )
        obs_all = np.concatenate(self.obs_list, axis=0)
        mask_all = np.concatenate(self.mask_list, axis=0)
        policy_all = np.concatenate(self.policy_list, axis=0)
        value_all = np.concatenate(self.value_list, axis=0)
        reason_all = np.concatenate(self.reason_list, axis=0)
        return CompactBatch(
            obs=obs_all,
            mask=mask_all,
            target_policy=policy_all,
            value=value_all,
            reason=reason_all,
        )

    def clear(self) -> None:
        self.obs_list.clear()
        self.mask_list.clear()
        self.policy_list.clear()
        self.value_list.clear()
        self.reason_list.clear()
        self.total_samples = 0

    def __len__(self) -> int:
        return self.total_samples
