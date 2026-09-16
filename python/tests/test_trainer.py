"""Tests for CompactDataset, ShardedBuffer, and Trainer."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from splendor_ai import (
    CompactBatch,
    CompactDataset,
    ShardedBuffer,
    SplendorNet,
    Trainer,
    TrainerConfig,
    generate_heuristic_compact_batch,
)


def test_heuristic_compact_generation():
    batch = generate_heuristic_compact_batch(num_games=5, start_seed=42)
    assert batch.num_samples > 0
    assert batch.obs.shape == (batch.num_samples, SplendorNet.OBS_SIZE)
    assert batch.mask.shape == (batch.num_samples, SplendorNet.ACTION_SIZE)
    assert batch.target_policy.shape == (batch.num_samples, SplendorNet.ACTION_SIZE)
    assert batch.action.shape == (batch.num_samples,)
    assert batch.value.shape == (batch.num_samples, 2)
    assert batch.reason.shape == (batch.num_samples, 3)
    assert (batch.action >= 0).all() and (batch.action < SplendorNet.ACTION_SIZE).all()


def test_sharded_buffer(tmp_path: Path):
    buffer = ShardedBuffer(shard_dir=tmp_path / "shards")
    batch = generate_heuristic_compact_batch(num_games=2, start_seed=123)

    saved_path = buffer.add_shard(batch)
    assert saved_path.exists()
    assert len(buffer.shard_files) == 1

    # 验证加载
    loaded_shards = list(buffer.iter_shards())
    assert len(loaded_shards) == 1
    assert loaded_shards[0].num_samples == batch.num_samples
    assert np.array_equal(loaded_shards[0].action, batch.action)


def test_trainer_with_compact_dataset(tmp_path: Path):
    batch = generate_heuristic_compact_batch(num_games=5, start_seed=200)
    dataset = CompactDataset(batch)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    net = SplendorNet(
        spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64
    )
    cfg = TrainerConfig(
        device="cpu",
        batch_size=32,
        ckpt_dir=str(tmp_path),
        t_max_epochs=2,
    )
    trainer = Trainer(net, cfg)

    m1 = trainer.train_epoch(loader)
    m2 = trainer.train_epoch(loader)

    assert "loss" in m1 and not np.isnan(m1["loss"])
    assert "top1_acc" in m1 and 0.0 <= m1["top1_acc"] <= 1.0
    assert trainer.epoch == 2

    # 测试检查点
    ckpt_path = trainer.save_checkpoint(
        "compact_test.pt",
        meta={"total_games": 10, "total_samples": batch.num_samples, "iteration": 1},
    )
    assert ckpt_path.exists()

    # 测试加载与元数据保持
    net2 = SplendorNet(
        spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64
    )
    trainer2 = Trainer(net2, cfg)
    meta = trainer2.load_checkpoint(ckpt_path)
    assert trainer2.epoch == 2
    assert meta.get("total_games") == 10
    assert meta.get("total_samples") == batch.num_samples
