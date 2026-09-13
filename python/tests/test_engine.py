"""Tests for splendor_ai engine and SplendorDuelEnv."""

import random
import numpy as np
import pytest

from splendor_ai import PyGameState, SplendorDuelEnv


def test_py_game_state_basic():
    game = PyGameState(42)
    obs = game.observe()
    mask = game.action_mask()
    legals = game.legal_action_ids()

    assert len(obs) == 726
    assert len(mask) == 288
    assert len(legals) >= 1
    assert game.current_player() in [0, 1]
    assert game.turn_number() == 1
    assert game.round_number() == 1
    assert not game.is_done()
    assert game.winner() is None


def test_env_reset_and_info():
    env = SplendorDuelEnv(seed=123)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (726,)
    assert obs.dtype == np.float32
    assert "action_mask" in info
    assert "legal_actions" in info
    assert "round_number" in info
    assert info["round_number"] == 1
    assert env.round_number == 1
    assert info["action_mask"].shape == (288,)
    assert len(info["legal_actions"]) > 0


def test_env_random_playout():
    env = SplendorDuelEnv(seed=2026)
    obs, info = env.reset()
    random.seed(2026)

    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        steps += 1
        legals = info["legal_actions"]
        assert len(legals) > 0, "Non-terminal state must have legal actions"

        action = random.choice(legals)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (726,)
        assert not np.isnan(obs).any()
        assert (obs >= 0.0).all() and (obs <= 1.0001).all()

    assert terminated, "Game should terminate within reasonable steps"
    assert info["winner"] in [0, 1]
    assert steps < 1000
    print(f"Random playout finished in {steps} steps, winner: {info['winner']}")


def test_env_cloning_independence():
    env = SplendorDuelEnv(seed=42)
    env.reset()

    # 推进 5 步
    for _ in range(5):
        legals = env.legal_actions
        env.step(legals[0])

    clone_env = env.clone()

    # 验证克隆后的状态完全一致
    assert np.array_equal(env.action_mask, clone_env.action_mask)
    assert env.current_player == clone_env.current_player

    # 分别走不同分支
    legals = env.legal_actions
    if len(legals) >= 2:
        env.step(legals[0])
        clone_env.step(legals[1])
        assert not np.array_equal(env.game.observe(), clone_env.game.observe())
