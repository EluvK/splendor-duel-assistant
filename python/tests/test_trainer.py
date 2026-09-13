"""Tests for Trainer and Dataset/Self-play pipelines."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from splendor_ai import (
    ReplayBuffer,
    Sample,
    SplendorDataset,
    SplendorNet,
    Trainer,
    TrainerConfig,
    generate_heuristic_dataset,
    generate_selfplay_dataset,
)


def test_heuristic_dataset_generation():
    samples = generate_heuristic_dataset(num_games=2, start_seed=42)
    assert len(samples) > 0

    s = samples[0]
    assert isinstance(s, Sample)
    assert s.obs.shape == (725,)
    assert s.mask.shape == (256,)
    assert s.target_policy.shape == (256,)
    assert s.target_value in [-1.0, 1.0]
    assert np.isclose(s.target_policy.sum(), 1.0)


def test_replay_buffer_capacity():
    buf = ReplayBuffer(max_samples=10)
    samples = [
        Sample(
            obs=np.zeros(725, dtype=np.float32),
            mask=np.zeros(256, dtype=bool),
            target_policy=np.zeros(256, dtype=np.float32),
            target_value=1.0,
        )
        for _ in range(15)
    ]
    buf.add_samples(samples)
    assert len(buf) == 10


def test_trainer_single_epoch(tmp_path: Path):
    samples = generate_heuristic_dataset(num_games=3, start_seed=100)
    dataset = SplendorDataset(samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    net = SplendorNet(spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64)
    cfg = TrainerConfig(
        device="cpu",
        batch_size=16,
        ckpt_dir=str(tmp_path),
        t_max_epochs=2,
    )
    trainer = Trainer(net, cfg)

    # 运行 2 个 Epoch
    m1 = trainer.train_epoch(loader)
    m2 = trainer.train_epoch(loader)

    assert "loss" in m1 and not np.isnan(m1["loss"])
    assert "top1_acc" in m1 and 0.0 <= m1["top1_acc"] <= 1.0
    assert trainer.epoch == 2

    # 测试检查点保存与加载
    ckpt_path = trainer.save_checkpoint("test_ckpt.pt")
    assert ckpt_path.exists()

    new_net = SplendorNet(spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64)
    new_trainer = Trainer(new_net, cfg)
    new_trainer.load_checkpoint(ckpt_path)
    assert new_trainer.epoch == 2


def test_selfplay_dataset_generation():
    net = SplendorNet(spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64)
    device = torch.device("cpu")
    samples = generate_selfplay_dataset(net, device, num_games=1, start_seed=42)
    assert len(samples) > 0
    assert samples[0].obs.shape == (725,)
    assert samples[0].target_value in [-1.0, 1.0]
