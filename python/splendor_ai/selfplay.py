"""Self-play and heuristic dataset generators for Splendor Duel."""

from typing import List, Optional, Tuple
import numpy as np
import torch

from splendor_ai._engine import (
    generate_heuristic_samples,
    generate_neural_mcts_samples,
)
from splendor_ai.dataset import CompactBatch
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet

# 回合时间衰减折现因子：统一默认 0.98
DEFAULT_GAMMA_TURN: float = 0.98


def generate_heuristic_compact_batch(
    num_games: int, start_seed: int = 42
) -> CompactBatch:
    """全速调用底层 Rust 8 线程并行模拟，生成连续紧凑样本块 (吞吐 > 50 万步/秒)."""
    raw_obs, raw_masks, raw_policies, raw_values, raw_reasons, total_steps = generate_heuristic_samples(
        num_games, start_seed
    )

    obs = np.asarray(raw_obs, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.OBS_SIZE)
    masks = np.asarray(raw_masks, dtype=np.uint8).view(bool).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    target_policy = np.asarray(raw_policies, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    values = np.asarray(raw_values, dtype=np.float32).reshape(total_steps, 2)
    reasons = np.asarray(raw_reasons, dtype=np.float32).reshape(total_steps, 3)

    return CompactBatch(obs=obs, mask=masks, target_policy=target_policy, value=values, reason=reasons)


def generate_rust_neural_mcts_compact_batch(
    net: Optional[SplendorNet] = None,
    onnx_bytes: Optional[bytes] = None,
    num_games: int = 100,
    num_simulations: int = 30,
    start_seed: int = 42,
    temp_steps: int = 12,
    temp_final: float = 0.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> CompactBatch:
    """全速调用底层 Rust 8 线程并行 ONNX 纯神经网络 MCTS，秒级产出正统 AlphaZero 自博弈样本."""
    if onnx_bytes is None:
        if net is None:
            raise ValueError("Either net or onnx_bytes must be provided")
        onnx_bytes = net.export_onnx_bytes()

    raw_obs, raw_masks, raw_policies, raw_values, raw_reasons, total_steps = generate_neural_mcts_samples(
        onnx_bytes,
        num_games,
        num_simulations,
        start_seed,
        temp_steps,
        temp_final,
        dirichlet_alpha,
        dirichlet_eps,
    )

    obs = np.asarray(raw_obs, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.OBS_SIZE)
    masks = np.asarray(raw_masks, dtype=np.uint8).view(bool).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    target_policy = np.asarray(raw_policies, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    values = np.asarray(raw_values, dtype=np.float32).reshape(total_steps, 2)
    reasons = np.asarray(raw_reasons, dtype=np.float32).reshape(total_steps, 3)

    return CompactBatch(obs=obs, mask=masks, target_policy=target_policy, value=values, reason=reasons)


def generate_selfplay_compact_batch(
    net: SplendorNet,
    device: torch.device,
    num_games: int,
    start_seed: int = 42,
    temperature: float = 1.0,
    gamma_turn: float = DEFAULT_GAMMA_TURN,
) -> CompactBatch:
    """使用当前神经网络直觉概率进行快速自博弈 (不跑 MCTS，用于超快速粗糙探索)."""
    all_obs: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_policies: List[np.ndarray] = []
    all_values: List[float] = []
    all_reasons: List[float] = []

    net.eval()
    with torch.no_grad():
        for g in range(num_games):
            env = SplendorDuelEnv(seed=start_seed + g)
            obs, info = env.reset()

            raw_trajectory = []
            while not env.is_done:
                acting_player = env.current_player
                turn_num = env.turn_number
                mask = info["action_mask"]

                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

                probs, _, _, _ = net.predict_action_probs(obs_t, mask_t, temperature=temperature)
                probs_np = probs.cpu().numpy()[0]
                action = int(np.random.choice(len(probs_np), p=probs_np))

                raw_trajectory.append((obs, mask, probs_np, acting_player, turn_num))
                obs, reward, terminated, truncated, info = env.step(action)
                if truncated:
                    break

            winner = info.get("winner")
            final_turn = env.turn_number
            # 终局多标签胜因编码 [20_pts, 10_crowns, 10_color]
            reason_multi_hot = [0.0, 0.0, 0.0]
            if winner is not None:
                scores = env.game.scores()
                crowns = env.game.crowns()
                w_score = scores[winner]
                w_crowns = crowns[winner]
                if w_score >= 20:
                    reason_multi_hot[0] = 1.0
                if w_crowns >= 10:
                    reason_multi_hot[1] = 1.0
                reason_str = str(info.get("reason", ""))
                if "TenPointsSameColor" in reason_str:
                    reason_multi_hot[2] = 1.0

            for obs_s, mask_s, pol_s, ply_s, turn_s in raw_trajectory:
                all_obs.append(obs_s)
                all_masks.append(mask_s)
                all_policies.append(pol_s)
                # 超时判双败 (-1.0)
                win_target = 1.0 if ply_s == winner else (-1.0 if winner is not None else -1.0)
                rem_turns = max(0, final_turn - turn_s)
                turns_target = min(rem_turns / 80.0, 1.0)
                all_values.extend([win_target, turns_target])
                all_reasons.extend(reason_multi_hot)

    total_steps = len(all_policies)
    obs_arr = np.array(all_obs, dtype=np.float32) if total_steps > 0 else np.zeros((0, SplendorDuelEnv.OBS_SIZE), dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=bool) if total_steps > 0 else np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=bool)
    policies_arr = np.array(all_policies, dtype=np.float32) if total_steps > 0 else np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32)
    values_arr = np.array(all_values, dtype=np.float32).reshape(-1, 2) if total_steps > 0 else np.zeros((0, 2), dtype=np.float32)
    reasons_arr = np.array(all_reasons, dtype=np.float32).reshape(-1, 3) if total_steps > 0 else np.zeros((0, 3), dtype=np.float32)

    return CompactBatch(obs=obs_arr, mask=masks_arr, target_policy=policies_arr, value=values_arr, reason=reasons_arr)
