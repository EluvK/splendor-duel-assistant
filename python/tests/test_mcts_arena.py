"""Tests for agents and Arena evaluation."""

import numpy as np
import pytest
import torch

from splendor_ai import (
    Arena,
    HeuristicAgent,
    PolicyNetAgent,
    RandomAgent,
    SplendorDuelEnv,
    SplendorNet,
)


def test_heuristic_agent_action():
    env = SplendorDuelEnv(seed=42)
    env.reset()

    agent = HeuristicAgent()
    action = agent.select_action(env)

    assert isinstance(action, int)
    assert action in env.legal_actions


def test_arena_paired_match():
    agent0 = RandomAgent()
    agent1 = RandomAgent()

    arena = Arena(agent0, agent1, agent0_name="Rand0", agent1_name="Rand1")
    res = arena.play_match(num_pairs=2, base_seed=123)

    assert res.total_games == 4
    assert res.agent0_wins + res.agent1_wins + res.draws == 4
    assert res.avg_steps > 0


def test_heuristic_vs_random_arena():
    agent_heu = HeuristicAgent()
    agent_rand = RandomAgent()

    arena = Arena(agent_heu, agent_rand, agent0_name="Heuristic", agent1_name="Rand")
    res = arena.play_match(num_pairs=2, base_seed=999)

    assert res.total_games == 4
    # 启发式规则 AI 对战纯随机 AI 应该取得压倒性胜利
    assert res.agent0_wins >= 3
