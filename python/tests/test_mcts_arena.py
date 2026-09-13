"""Tests for MCTS tree search and Arena evaluation."""

import numpy as np
import pytest
import torch

from splendor_ai import (
    Arena,
    HeuristicAgent,
    MCTS,
    MCTSAgent,
    PolicyNetAgent,
    RandomAgent,
    SplendorDuelEnv,
    SplendorNet,
)


def test_mcts_search_basic():
    net = SplendorNet(spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64)
    device = torch.device("cpu")
    mcts = MCTS(net, device, c_puct=1.5)

    env = SplendorDuelEnv(seed=42)
    obs, info = env.reset()

    pi, action = mcts.search(env, num_simulations=20, add_noise=True, temperature=1.0)

    assert isinstance(pi, np.ndarray)
    assert pi.shape == (256,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert action in env.legal_actions


def test_arena_paired_match():
    # 测试成对种子对战
    agent0 = RandomAgent()
    agent1 = RandomAgent()

    arena = Arena(agent0, agent1, agent0_name="Rand0", agent1_name="Rand1")
    res = arena.play_match(num_pairs=2, base_seed=123)

    assert res.total_games == 4
    assert res.agent0_wins + res.agent1_wins + res.draws == 4
    assert res.avg_steps > 0


def test_mcts_agent_play():
    net = SplendorNet(spatial_channels=16, num_res_blocks=1, context_hidden=64, fusion_hidden=64)
    device = torch.device("cpu")
    agent_mcts = MCTSAgent(net, device, num_sims=5, temperature=0.0)
    agent_rand = RandomAgent()

    arena = Arena(agent_mcts, agent_rand, agent0_name="MCTS", agent1_name="Rand")
    res = arena.play_match(num_pairs=1, base_seed=999)

    assert res.total_games == 2
