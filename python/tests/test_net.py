"""Tests for SplendorNet neural network and HeuristicAI evaluation."""

import random
import numpy as np
import pytest
import torch

from splendor_ai import SplendorDuelEnv, SplendorNet


def test_net_forward_shapes():
    net = SplendorNet(spatial_channels=32, num_res_blocks=2)

    # 单样本前向
    single_obs = torch.randn(726)
    logits, value = net(single_obs)
    assert logits.shape == (1, 288)
    assert value.shape == (1, 1)
    assert (-1.0 <= value.item() <= 1.0)

    # 批处理前向 (Batch Size = 8)
    batch_obs = torch.randn(8, 726)
    batch_logits, batch_value = net(batch_obs)
    assert batch_logits.shape == (8, 288)
    assert batch_value.shape == (8, 1)
    assert (batch_value >= -1.0).all() and (batch_value <= 1.0).all()


def test_net_action_masking():
    net = SplendorNet()
    env = SplendorDuelEnv(seed=42)
    obs, info = env.reset()

    obs_t = torch.from_numpy(obs)
    mask_t = torch.from_numpy(info["action_mask"])

    probs, value = net.predict_action_probs(obs_t, mask_t)

    assert probs.shape == (1, 288)
    probs_np = probs.detach().cpu().numpy()[0]

    # 验证非法动作的概率为 0
    illegal_indices = np.where(~info["action_mask"])[0]
    legal_indices = np.where(info["action_mask"])[0]

    assert len(legal_indices) > 0
    assert np.allclose(probs_np[illegal_indices], 0.0, atol=1e-6)
    assert np.isclose(probs_np[legal_indices].sum(), 1.0, atol=1e-5)


def test_heuristic_ai_vs_random():
    """验证启发式 AI 对抗纯随机 AI 的碾压优势."""
    env = SplendorDuelEnv()
    num_games = 10
    heuristic_wins = 0
    total_steps = []

    for seed in range(num_games):
        obs, info = env.reset(seed=seed)
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            steps += 1
            curr_player = env.current_player

            if curr_player == 0:
                # Player 0: 启发式 AI
                action = env.heuristic_action(seed=seed + steps)
                assert action is not None
            else:
                # Player 1: 纯随机 AI
                action = random.choice(info["legal_actions"])

            obs, reward, terminated, truncated, info = env.step(action)

        winner = info["winner"]
        if winner == 0:
            heuristic_wins += 1
        total_steps.append(steps)

    win_rate = heuristic_wins / num_games
    avg_steps = sum(total_steps) / num_games
    print(f"\n启发式 AI 对战纯随机 AI: 胜率 = {win_rate * 100:.1f}%, 平均步数 = {avg_steps:.1f}")

    # 启发式 AI 应展现压倒性优势 (胜率 >= 90%)
    assert win_rate >= 0.9, f"启发式 AI 胜率过低: {win_rate}"
    # 相比双随机 AI 对决的 340+ 步，单方加入启发式 AI 应能显著提速终局
    assert avg_steps < 250, f"启发式 AI 耗时过长: {avg_steps} 步"
