"""Compact and sharded dataset implementations for Splendor Duel."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CompactBatch:
    """紧凑内存样本块 (零多余对象开销，纯连续扁平数组)."""

    obs: np.ndarray  # [N, 725] float32
    mask: np.ndarray  # [N, 256] bool
    action: np.ndarray  # [N] int64 (标量整数动作 ID，彻底废除 256 维浮点 One-Hot 浪费)
    value: np.ndarray  # [N, 1] float32

    @property
    def num_samples(self) -> int:
        return len(self.action)

    def save_npz(self, path: Path) -> None:
        """持久化保存为紧凑压缩分片文件."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
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


class CompactDataset(Dataset):
    """基于紧凑连续内存数组的 PyTorch 数据集."""

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
        self.shard_files: List[Path] = sorted(list(self.shard_dir.glob("shard_*.npz")))

    def add_shard(self, batch: CompactBatch) -> Path:
        """保存新分片."""
        shard_idx = len(self.shard_files) + 1
        path = self.shard_dir / f"shard_{shard_idx:05d}.npz"
        batch.save_npz(path)
        self.shard_files.append(path)
        return path

    def iter_shards(self) -> Iterator[CompactBatch]:
        """流式遍历所有分片，训练完一个自动释放内存，内存永远恒定."""
        for p in self.shard_files:
            yield CompactBatch.load_npz(p)

    def clear(self) -> None:
        for p in self.shard_files:
            if p.exists():
                p.unlink()
        self.shard_files.clear()
