"""Tests for Rust Neural-MCTS, ReplayBuffer, and Neural SelfPlay."""

import numpy as np
import pytest
import torch

from splendor_ai import (
    CompactBatch,
    ReplayBuffer,
    SplendorDuelEnv,
    SplendorNet,
)
from splendor_ai.selfplay import generate_rust_neural_mcts_compact_batch


def test_replay_buffer_sliding_window():
    buffer = ReplayBuffer(max_samples=100)

    # 创建伪 batch 1 (60 样本)
    b1 = CompactBatch(
        obs=np.ones((60, SplendorDuelEnv.OBS_SIZE), dtype=np.float32),
        mask=np.ones((60, SplendorDuelEnv.ACTION_SIZE), dtype=bool),
        action=np.zeros(60, dtype=np.int64),
        value=np.ones((60, 2), dtype=np.float32),
        reason=np.zeros((60, 3), dtype=np.float32),
    )
    buffer.add_batch(b1)
    assert len(buffer) == 60

    # 创建伪 batch 2 (70 样本) -> 触发滑动淘汰，总容量维持在合理区间
    b2 = CompactBatch(
        obs=np.full((70, SplendorDuelEnv.OBS_SIZE), 2.0, dtype=np.float32),
        mask=np.ones((70, SplendorDuelEnv.ACTION_SIZE), dtype=bool),
        action=np.ones(70, dtype=np.int64),
        value=np.full((70, 2), -1.0, dtype=np.float32),
        reason=np.ones((70, 3), dtype=np.float32),
    )
    buffer.add_batch(b2)
    assert len(buffer) == 70  # b1 被滑出，保留 b2

    out_batch = buffer.get_compact_batch()
    assert out_batch.num_samples == 70
    assert np.all(out_batch.obs == 2.0)
    assert out_batch.reason.shape == (70, 6)


def test_generate_rust_neural_mcts_selfplay():
    torch.manual_seed(42)
    net = SplendorNet()
    batch = generate_rust_neural_mcts_compact_batch(
        net=net,
        num_games=2,
        num_simulations=10,
        start_seed=42,
        temp_steps=4,
    )

    assert batch.num_samples > 0
    assert batch.obs.shape[1] == SplendorDuelEnv.OBS_SIZE
    assert batch.mask.shape[1] == SplendorDuelEnv.ACTION_SIZE
    assert batch.target_policy.shape == (batch.num_samples, SplendorDuelEnv.ACTION_SIZE)
    assert len(batch.action) == batch.num_samples
    assert batch.value.shape == (batch.num_samples, 2)
    assert batch.reason.shape == (batch.num_samples, 6)
    # 验证解耦胜率预期 [-1.0, 1.0] 与归一化剩余步数 [0.0, 1.0]
    assert np.all(batch.value[:, 0] >= -1.0) and np.all(batch.value[:, 0] <= 1.0)
    assert np.all(batch.value[:, 1] >= 0.0) and np.all(batch.value[:, 1] <= 1.0)
    assert not np.isnan(batch.value).any()
    assert np.all(batch.reason >= 0.0) and np.all(batch.reason <= 1.0)


def test_evaluate_neural_match_with_mcts():
    from splendor_ai._engine import evaluate_neural_match

    torch.manual_seed(42)
    net = SplendorNet()
    bytes0 = net.export_onnx_bytes()

    # 1. 神经模型 vs 启发式 AI (带 MCTS)
    total, a0, a1, draws, reasons = evaluate_neural_match(
        bytes0, None, num_pairs=1, base_seed=42, num_sims=5
    )
    assert total == 2
    assert a0 + a1 + draws == 2
    assert "p0_seat_wins" in reasons

    # 2. 神经模型 vs 神经模型 (带 MCTS)
    total, a0, a1, draws, reasons = evaluate_neural_match(
        bytes0, bytes0, num_pairs=1, base_seed=42, num_sims=5
    )
    assert total == 2
    assert a0 + a1 + draws == 2


