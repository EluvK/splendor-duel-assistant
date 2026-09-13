"""Compact, vectorized, and sharded dataset implementations for Splendor Duel."""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CompactBatch:
    """紧凑内存样本块 (零多余对象开销，纯连续扁平数组)."""

    obs: np.ndarray  # [N, 725] float32
    mask: np.ndarray  # [N, 256] bool
    action: np.ndarray  # [N] int64 (标量整数动作 ID)
    value: np.ndarray  # [N, 1] float32

    @property
    def num_samples(self) -> int:
        return len(self.action)

    def save_npz(self, path: Path, compressed: bool = False) -> None:
        """持久化保存为分片文件 (默认未压缩以取得最大读写吞吐)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_fn = np.savez_compressed if compressed else np.savez
        save_fn(
            path,
            obs=self.obs,
            mask=self.mask,
            action=self.action,
            value=self.value,
        )

    @classmethod
    def load_npz(cls, path: Path) -> "CompactBatch":
        """从分片文件加载样本."""
        data = np.load(path)
        return cls(
            obs=data["obs"],
            mask=data["mask"],
            action=data["action"],
            value=data["value"],
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
        self.action = torch.from_numpy(batch.action).long()
        self.value = torch.from_numpy(batch.value).float()

        self.resident_on_device = False
        if device is not None and device.type == "cuda":
            try:
                # 单个分片通常约几百MB，直接常驻显存，零总线传输延迟
                self.obs = self.obs.to(device)
                self.mask = self.mask.to(device)
                self.action = self.action.to(device)
                self.value = self.value.to(device)
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
            b_action = self.action[idx]
            b_value = self.value[idx]

            if not self.resident_on_device and self.device is not None:
                b_obs = b_obs.to(self.device, non_blocking=True)
                b_mask = b_mask.to(self.device, non_blocking=True)
                b_action = b_action.to(self.device, non_blocking=True)
                b_value = b_value.to(self.device, non_blocking=True)

            yield {
                "obs": b_obs,
                "mask": b_mask,
                "action": b_action,
                "value": b_value,
            }


class CompactDataset(Dataset):
    """基于紧凑连续内存数组的标准 PyTorch 数据集 (兼容传统 DataLoader)."""

    def __init__(self, batch: CompactBatch) -> None:
        self.obs = torch.from_numpy(batch.obs).float()
        self.mask = torch.from_numpy(batch.mask).bool()
        self.action = torch.from_numpy(batch.action).long()
        self.value = torch.from_numpy(batch.value).float()

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.obs[idx],
            "mask": self.mask[idx],
            "action": self.action[idx],
            "value": self.value[idx],
        }


class ShardedBuffer:
    """面向百万局级别训练的磁盘分片缓存管理器."""

    def __init__(self, shard_dir: Path) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.refresh()

    def refresh(self) -> None:
        self.shard_files: List[Path] = sorted(list(self.shard_dir.glob("shard_*.npz")))

    def add_shard(self, batch: CompactBatch, compressed: bool = False) -> Path:
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
        """快速统计所有分片的总样本数 (仅读 action 长度)."""
        total = 0
        for p in self.shard_files:
            with np.load(p) as data:
                total += len(data["action"])
        return total

    def clear(self) -> None:
        """清空所有分片."""
        for p in self.shard_files:
            if p.exists():
                p.unlink()
        self.shard_files.clear()
