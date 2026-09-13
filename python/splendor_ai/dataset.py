"""Dataset and Replay Buffer definitions for Splendor Duel."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    """单个决策步样本."""

    obs: np.ndarray  # [725] float32 状态观察向量
    mask: np.ndarray  # [256] bool 合法动作掩码
    target_policy: np.ndarray  # [256] float32 动作概率分布
    target_value: float  # [-1.0, 1.0] 行动方最终胜负结果 (+1 胜 / -1 负)


class SplendorDataset(Dataset):
    """PyTorch 对局数据集."""

    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "obs": torch.from_numpy(item.obs).float(),
            "mask": torch.from_numpy(item.mask).bool(),
            "target_policy": torch.from_numpy(item.target_policy).float(),
            "target_value": torch.tensor([item.target_value], dtype=torch.float32),
        }


class ReplayBuffer:
    """经验回放缓存池，维护滚动对局样本."""

    def __init__(self, max_samples: int = 200_000) -> None:
        self.max_samples = max_samples
        self.samples: List[Sample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def add_sample(self, sample: Sample) -> None:
        if len(self.samples) >= self.max_samples:
            self.samples.pop(0)
        self.samples.append(sample)

    def add_samples(self, new_samples: List[Sample]) -> None:
        if not new_samples:
            return
        # 若新增样本超出了容量，只截取末尾
        if len(new_samples) >= self.max_samples:
            self.samples = new_samples[-self.max_samples :]
            return

        overflow = (len(self.samples) + len(new_samples)) - self.max_samples
        if overflow > 0:
            self.samples = self.samples[overflow:]
        self.samples.extend(new_samples)

    def to_dataset(self) -> SplendorDataset:
        return SplendorDataset(self.samples)

    def clear(self) -> None:
        self.samples.clear()
