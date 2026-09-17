"""Self-play and heuristic dataset generators for Splendor Duel."""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from splendor_ai._engine import (
    evaluate_gpu_batched_neural_match,
    generate_gpu_batched_neural_mcts_samples,
    generate_heuristic_samples,
    generate_neural_mcts_samples,
    generate_neural_mcts_match_samples,
)
from splendor_ai.dataset import CompactBatch
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.gpu_worker import GpuBatchedEvaluator
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


def generate_rust_neural_mcts_match_compact_batch(
    net: Optional[SplendorNet] = None,
    onnx_bytes: Optional[bytes] = None,
    opp_net: Optional[SplendorNet] = None,
    opp_onnx_bytes: Optional[bytes] = None,
    num_games: int = 20,
    num_simulations: int = 30,
    start_seed: int = 42,
    temp_steps: int = 12,
    temp_final: float = 0.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    record_opponent: bool = False,
) -> CompactBatch:
    """全速调用底层 Rust 进行主模型与指定对手 (历史模型或启发式 AI) 的严格换座对抗采样."""
    if onnx_bytes is None:
        if net is None:
            raise ValueError("Either net or onnx_bytes must be provided for primary agent")
        onnx_bytes = net.export_onnx_bytes()

    if opp_onnx_bytes is None and opp_net is not None:
        opp_onnx_bytes = opp_net.export_onnx_bytes()

    raw_obs, raw_masks, raw_policies, raw_values, raw_reasons, total_steps = generate_neural_mcts_match_samples(
        onnx_bytes,
        opp_onnx_bytes,
        num_games,
        num_simulations,
        start_seed,
        temp_steps,
        temp_final,
        dirichlet_alpha,
        dirichlet_eps,
        record_opponent,
    )

    obs = np.asarray(raw_obs, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.OBS_SIZE)
    masks = np.asarray(raw_masks, dtype=np.uint8).view(bool).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    target_policy = np.asarray(raw_policies, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    values = np.asarray(raw_values, dtype=np.float32).reshape(total_steps, 2)
    reasons = np.asarray(raw_reasons, dtype=np.float32).reshape(total_steps, 3)

    return CompactBatch(obs=obs, mask=masks, target_policy=target_policy, value=values, reason=reasons)


def generate_gpu_batched_mcts_compact_batch(
    net: SplendorNet,
    num_games: int = 100,
    num_simulations: int = 30,
    max_concurrent_games: int = 128,
    start_seed: int = 42,
    temp_steps: int = 12,
    temp_final: float = 0.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    heuristic_ratio: float = 0.0,
    record_opponent: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> CompactBatch:
    """调用底层 Rust 向量化批推演引擎与 PyTorch GPU 批评估，全速产出高质量 AlphaZero 自博弈样本."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluator = GpuBatchedEvaluator(model=net, device=device)

    raw_obs, raw_masks, raw_policies, raw_values, raw_reasons, total_steps = generate_gpu_batched_neural_mcts_samples(
        evaluator,
        num_games,
        num_simulations,
        max_concurrent_games,
        start_seed,
        temp_steps,
        temp_final,
        dirichlet_alpha,
        dirichlet_eps,
        heuristic_ratio,
        record_opponent,
    )

    obs = np.asarray(raw_obs, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.OBS_SIZE)
    masks = np.asarray(raw_masks, dtype=np.uint8).view(bool).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    target_policy = np.asarray(raw_policies, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    values = np.asarray(raw_values, dtype=np.float32).reshape(total_steps, 2)
    reasons = np.asarray(raw_reasons, dtype=np.float32).reshape(total_steps, 3)

    return CompactBatch(obs=obs, mask=masks, target_policy=target_policy, value=values, reason=reasons)


def evaluate_gpu_neural_match(
    net_c: SplendorNet,
    net_b: Optional[SplendorNet] = None,
    num_pairs: int = 30,
    base_seed: int = 1000,
    num_sims: int = 60,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[int, int, int, int, dict]:
    """使用 GPU 批推演执行严格成对换座的模型对抗评测 (支持双网络对决或候选网络对决启发式)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_cb_0 = GpuBatchedEvaluator(model=net_c, device=device)
    eval_cb_1 = GpuBatchedEvaluator(model=net_b, device=device) if net_b is not None else None

    return evaluate_gpu_batched_neural_match(
        eval_cb_0,
        eval_cb_1,
        num_pairs=num_pairs,
        base_seed=base_seed,
        num_sims=num_sims,
    )


def concat_compact_batches(batches: List[CompactBatch]) -> CompactBatch:
    """将多个紧凑批次高效拼接为一个统一批次 (零多余拷贝)."""
    valid_b = [b for b in batches if b.num_samples > 0]
    if not valid_b:
        return CompactBatch(
            obs=np.zeros((0, SplendorDuelEnv.OBS_SIZE), dtype=np.float32),
            mask=np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=bool),
            value=np.zeros((0, 2), dtype=np.float32),
            reason=np.zeros((0, 3), dtype=np.float32),
            target_policy=np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=np.float32),
        )
    if len(valid_b) == 1:
        return valid_b[0]

    obs = np.concatenate([b.obs for b in valid_b], axis=0)
    mask = np.concatenate([b.mask for b in valid_b], axis=0)
    target_policy = np.concatenate([b.target_policy for b in valid_b], axis=0)
    value = np.concatenate([b.value for b in valid_b], axis=0)
    reason = np.concatenate([b.reason for b in valid_b], axis=0)
    return CompactBatch(obs=obs, mask=mask, target_policy=target_policy, value=value, reason=reason)


def generate_league_mcts_compact_batch(
    net: Optional[SplendorNet] = None,
    onnx_bytes: Optional[bytes] = None,
    history_bytes_pool: Optional[List[bytes]] = None,
    total_games: int = 50,
    heuristic_ratio: float = 0.15,
    history_ratio: float = 0.15,
    num_simulations: int = 30,
    start_seed: int = 42,
    temp_steps: int = 12,
    temp_final: float = 0.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    record_opponent: bool = False,
) -> CompactBatch:
    """生成包含纯自博弈、启发式对抗和历史模型对抗的多元联赛样本 (彻底避免策略空间塌缩)."""
    if onnx_bytes is None:
        if net is None:
            raise ValueError("Either net or onnx_bytes must be provided")
        onnx_bytes = net.export_onnx_bytes()

    # 计算配比 (均规整为偶数以实现严格换座)
    n_heu = max(0, int(total_games * heuristic_ratio))
    n_heu = (n_heu // 2) * 2

    has_history = history_bytes_pool is not None and len(history_bytes_pool) > 0
    n_hist = max(0, int(total_games * history_ratio)) if has_history else 0
    n_hist = (n_hist // 2) * 2

    n_self = max(2, total_games - n_heu - n_hist)
    n_self = (n_self // 2) * 2

    batches: List[CompactBatch] = []
    curr_seed = start_seed

    # 1. 纯自博弈批次
    b_self = generate_rust_neural_mcts_compact_batch(
        onnx_bytes=onnx_bytes,
        num_games=n_self,
        num_simulations=num_simulations,
        start_seed=curr_seed,
        temp_steps=temp_steps,
        temp_final=temp_final,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
    )
    batches.append(b_self)
    curr_seed += n_self * 1009

    # 2. 启发式对抗批次
    if n_heu > 0:
        b_heu = generate_rust_neural_mcts_match_compact_batch(
            onnx_bytes=onnx_bytes,
            opp_onnx_bytes=None,  # 对手为 HeuristicAI
            num_games=n_heu,
            num_simulations=num_simulations,
            start_seed=curr_seed,
            temp_steps=temp_steps,
            temp_final=temp_final,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            record_opponent=record_opponent,
        )
        batches.append(b_heu)
        curr_seed += n_heu * 1009

    # 3. 历史模型对抗批次
    if n_hist > 0 and has_history:
        import random
        opp_bytes = random.choice(history_bytes_pool)
        b_hist = generate_rust_neural_mcts_match_compact_batch(
            onnx_bytes=onnx_bytes,
            opp_onnx_bytes=opp_bytes,
            num_games=n_hist,
            num_simulations=num_simulations,
            start_seed=curr_seed,
            temp_steps=temp_steps,
            temp_final=temp_final,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            record_opponent=record_opponent,
        )
        batches.append(b_hist)

    return concat_compact_batches(batches)


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
