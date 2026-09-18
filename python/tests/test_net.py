"""Tests for SplendorNet neural network and HeuristicAI evaluation."""

import random
import numpy as np
import pytest
import torch

from splendor_ai import SplendorDuelEnv, SplendorNet


def test_net_forward_shapes():
    net = SplendorNet(spatial_channels=32, num_res_blocks=2)

    # 单样本前向
    single_obs = torch.randn(SplendorNet.OBS_SIZE)
    logits, win_v, turns_v, reason_logits = net(single_obs)
    assert logits.shape == (1, SplendorNet.ACTION_SIZE)
    assert win_v.shape == (1, 1)
    assert turns_v.shape == (1, 1)
    assert reason_logits.shape == (1, 6)
    assert (-1.0 <= win_v.item() <= 1.0)
    assert (0.0 <= turns_v.item() <= 1.0)

    # 批处理前向 (Batch Size = 8)
    batch_obs = torch.randn(8, SplendorNet.OBS_SIZE)
    b_logits, b_win, b_turns, b_reason = net(batch_obs)
    assert b_logits.shape == (8, SplendorNet.ACTION_SIZE)
    assert b_win.shape == (8, 1)
    assert b_turns.shape == (8, 1)
    assert b_reason.shape == (8, 6)
    assert (b_win >= -1.0).all() and (b_win <= 1.0).all()
    assert (b_turns >= 0.0).all() and (b_turns <= 1.0).all()


def test_net_action_masking():
    net = SplendorNet()
    env = SplendorDuelEnv(seed=42)
    obs, info = env.reset()

    obs_t = torch.from_numpy(obs)
    mask_t = torch.from_numpy(info["action_mask"])

    probs, win_v, turns_v, reason_logits = net.predict_action_probs(obs_t, mask_t)

    assert probs.shape == (1, SplendorNet.ACTION_SIZE)
    assert win_v.shape == (1, 1)
    assert turns_v.shape == (1, 1)
    assert reason_logits.shape == (1, 6)
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


def test_splendornet_v3_multi_task_loss_and_backward():
    """测试 SplendorNet 实体解耦策略头、二分类胜率与稠密辅助头的完整反向传播."""
    net = SplendorNet()
    obs = torch.randn(4, SplendorNet.OBS_SIZE)
    target_action = torch.tensor([10, 175, 550, 600], dtype=torch.long)
    target_win = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]], dtype=torch.float32)
    target_turns = torch.tensor([[0.2], [0.8], [0.4], [0.6]], dtype=torch.float32)
    target_reason = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    logits, win_v, turns_v, reason_logits, win_logits, lead_v = net.forward_train(obs)
    assert win_logits.shape == (4, 2)
    assert reason_logits.shape == (4, 6)
    policy_loss = torch.nn.functional.cross_entropy(logits, target_action)

    target_win_dist = SplendorNet.compute_win_target_distribution(target_win)
    win_loss = -(target_win_dist * torch.nn.functional.log_softmax(win_logits, dim=-1)).sum(dim=-1).mean()

    turns_loss = torch.nn.functional.smooth_l1_loss(turns_v, target_turns)
    reason_loss = torch.nn.functional.binary_cross_entropy_with_logits(reason_logits, target_reason)
    target_leads = SplendorNet.extract_state_leads(obs)
    lead_loss = torch.nn.functional.smooth_l1_loss(lead_v, target_leads)

    total_loss, weights = net.compute_multi_task_loss(
        policy_loss, win_loss, turns_loss, reason_loss, lead_loss
    )

    assert total_loss.item() > 0.0
    assert len(weights) == 5
    for k in ["w_policy", "w_win", "w_turns", "w_reason", "w_lead"]:
        assert weights[k] > 0.0

    total_loss.backward()
    assert net.reserve_gold_scorer[0].weight.grad is not None
    assert net.buy_card_scorer_c.weight.grad is not None
    assert net.plan_encoder[0].weight.grad is not None
    assert net.line_cell_scorer.weight.grad is not None
    assert net.card_encoder[0].weight.grad is not None
    assert net.lead_head[0].weight.grad is not None
    assert net.win_head[0].weight.grad is not None
    assert lead_v.shape == (4, 3)
    assert target_leads.shape == (4, 3)


def test_policy_head_scale_balance():
    """验证策略头在随机初始化时各动作分块的 Logits 尺度健康平衡."""
    torch.manual_seed(42)
    net = SplendorNet()
    obs = torch.randn(16, SplendorNet.OBS_SIZE)
    logits, _, _, _ = net(obs)

    # 预留 (172..547) 与购买 (547..1807)
    card_indices = list(range(172, 1807))
    discrete_indices = [i for i in range(SplendorNet.ACTION_SIZE) if i not in card_indices]

    card_logits = logits[:, card_indices]
    discrete_logits = logits[:, discrete_indices]

    card_std = card_logits.std().item()
    discrete_std = discrete_logits.std().item()

    # 验证标准差尺度比例处于健康范围
    ratio = card_std / discrete_std
    assert ratio < 5.0, f"卡牌与离散 logits 标准差比例过大: {ratio:.2f} (card={card_std:.2f}, discrete={discrete_std:.2f})"
    assert card_std < 2.5, f"卡牌 logits 标准差过大: {card_std:.2f}"

    # 在环境真实开局状态下，验证拿宝石动作先验概率不会被病态挤压
    env = SplendorDuelEnv(seed=42)
    obs_real, info = env.reset()
    # 开局若为 OptionalActions (仅动作0: 跳过)，执行步进进入 MandatoryAction 主动作阶段
    if info.get("phase") == "OptionalActions" and info["legal_actions"] == [0]:
        obs_real, _, _, _, info = env.step(0)

    probs, _, _, _ = net.predict_action_probs(
        torch.from_numpy(obs_real), torch.from_numpy(info["action_mask"])
    )
    probs_np = probs.detach().cpu().numpy()[0]

    # 开局拿宝石动作 (连线拿宝石位于 52..171)
    gem_line_actions = [a for a in info["legal_actions"] if 52 <= a <= 171]
    assert len(gem_line_actions) > 0
    gem_line_prob_sum = probs_np[gem_line_actions].sum()

    # 开局连线动作总概率应具有可观的探索空间 (不应被预留卡牌压制至 < 5%)
    assert gem_line_prob_sum > 0.05, f"开局连线拿宝石动作被过度压制: {gem_line_prob_sum:.4f}"


def test_compute_win_target_distribution():
    """测试二分类胜率目标分布计算的正确性与边界处理."""
    # 1. 验证极值点
    targets = torch.tensor([[1.0], [-1.0], [0.0]], dtype=torch.float32)
    dist = SplendorNet.compute_win_target_distribution(targets)
    assert dist.shape == (3, 2)
    # +1.0 对应 P(win)=1.0, P(loss)=0.0
    assert torch.allclose(dist[0], torch.tensor([1.0, 0.0]))
    # -1.0 对应 P(win)=0.0, P(loss)=1.0
    assert torch.allclose(dist[1], torch.tensor([0.0, 1.0]))
    # 0.0 对应 P(win)=0.5, P(loss)=0.5
    assert torch.allclose(dist[2], torch.tensor([0.5, 0.5]))

    # 2. 验证越界值 clamp
    oob = torch.tensor([[2.0], [-3.0]], dtype=torch.float32)
    dist_oob = SplendorNet.compute_win_target_distribution(oob)
    assert torch.allclose(dist_oob[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(dist_oob[1], torch.tensor([0.0, 1.0]))



