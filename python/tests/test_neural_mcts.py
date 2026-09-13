"""Tests for NeuralMCTS, ReplayBuffer, and Neural SelfPlay."""

import numpy as np
import pytest
import torch

from splendor_ai import (
    CompactBatch,
    NeuralMCTS,
    NeuralMCTSAgent,
    ReplayBuffer,
    SplendorDuelEnv,
    SplendorNet,
)
from splendor_ai.selfplay import (
    generate_neural_mcts_selfplay_compact_batch,
    generate_rust_neural_mcts_compact_batch,
)


def test_neural_mcts_search():
    device = torch.device("cpu")
    net = SplendorNet().to(device)
    mcts = NeuralMCTS(net, device, c_puct=1.5)

    env = SplendorDuelEnv(seed=101)
    env.reset()

    # 1. 确定性搜索 (温度 = 0.0, 无 Dirichlet 噪声)
    action, pi = mcts.search(env.game, num_sims=10, add_dirichlet=False, temperature=0.0)
    assert isinstance(action, int)
    assert action in env.legal_actions
    assert pi.shape == (SplendorDuelEnv.ACTION_SIZE,)
    assert np.isclose(pi.sum(), 1.0)
    # 确定性动作应对应访问概率最大者
    assert action == int(np.argmax(pi))

    # 2. 探索性搜索 (温度 = 1.0, 注入 Dirichlet 噪声)
    action_noisy, pi_noisy = mcts.search(
        env.game,
        num_sims=10,
        add_dirichlet=True,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        temperature=1.0,
    )
    assert isinstance(action_noisy, int)
    assert action_noisy in env.legal_actions
    assert np.isclose(pi_noisy.sum(), 1.0)


def test_neural_mcts_agent():
    device = torch.device("cpu")
    net = SplendorNet().to(device)
    agent = NeuralMCTSAgent(net, device, num_sims=10)

    env = SplendorDuelEnv(seed=202)
    env.reset()

    action = agent.select_action(env)
    assert isinstance(action, int)
    assert action in env.legal_actions


def test_replay_buffer_sliding_window():
    buffer = ReplayBuffer(max_samples=100)

    # 创建伪 batch 1 (60 样本)
    b1 = CompactBatch(
        obs=np.ones((60, 725), dtype=np.float32),
        mask=np.ones((60, 256), dtype=bool),
        action=np.zeros(60, dtype=np.int64),
        value=np.ones((60, 1), dtype=np.float32),
    )
    buffer.add_batch(b1)
    assert len(buffer) == 60

    # 创建伪 batch 2 (70 样本) -> 触发滑动淘汰，总容量维持在合理区间
    b2 = CompactBatch(
        obs=np.full((70, 725), 2.0, dtype=np.float32),
        mask=np.ones((70, 256), dtype=bool),
        action=np.ones(70, dtype=np.int64),
        value=np.full((70, 1), -1.0, dtype=np.float32),
    )
    buffer.add_batch(b2)
    assert len(buffer) == 70  # b1 被滑出，保留 b2

    out_batch = buffer.get_compact_batch()
    assert out_batch.num_samples == 70
    assert np.all(out_batch.obs == 2.0)


def test_generate_neural_mcts_selfplay():
    device = torch.device("cpu")
    net = SplendorNet().to(device)

    # 运行极简 1 局对弈、5 次模拟，验证端到端数据生成通畅
    batch = generate_neural_mcts_selfplay_compact_batch(
        net=net,
        device=device,
        num_games=1,
        num_simulations=5,
        start_seed=42,
        temp_threshold_steps=5,
    )

    assert batch.num_samples > 0
    assert batch.obs.shape[1] == SplendorDuelEnv.OBS_SIZE
    assert batch.mask.shape[1] == SplendorDuelEnv.ACTION_SIZE
    assert len(batch.action) == batch.num_samples
    assert batch.value.shape == (batch.num_samples, 1)
    # 胜负值应在 {-1.0, 1.0} 中


def test_generate_rust_neural_mcts_selfplay():
    net = SplendorNet()
    batch = generate_rust_neural_mcts_compact_batch(
        net=net,
        num_games=1,
        num_simulations=5,
        start_seed=123,
        temp_steps=4,
    )

    assert batch.num_samples > 0
    assert batch.obs.shape[1] == SplendorDuelEnv.OBS_SIZE
    assert batch.mask.shape[1] == SplendorDuelEnv.ACTION_SIZE
    assert len(batch.action) == batch.num_samples
    assert batch.value.shape == (batch.num_samples, 1)
    assert np.all(np.isin(batch.value, [-1.0, 1.0]))
    assert set(np.unique(batch.value)).issubset({-1.0, 1.0})
