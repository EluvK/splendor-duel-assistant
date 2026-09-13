"""Self-play and heuristic dataset generators for Splendor Duel."""

from typing import List, Tuple
import numpy as np
import torch

from splendor_ai._engine import generate_heuristic_samples, generate_mcts_samples
from splendor_ai.dataset import CompactBatch
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.mcts import NeuralMCTS
from splendor_ai.net import SplendorNet
from splendor_ai.progress import Progress


def generate_heuristic_compact_batch(
    num_games: int, start_seed: int = 42
) -> CompactBatch:
    """全速调用底层 Rust 8 线程并行模拟，生成连续紧凑样本块 (吞吐 > 50 万步/秒)."""
    raw_obs, raw_masks, raw_actions, raw_values, total_steps = generate_heuristic_samples(
        num_games, start_seed
    )

    obs = np.asarray(raw_obs, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.OBS_SIZE)
    masks = np.asarray(raw_masks, dtype=np.uint8).view(bool).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    actions = np.asarray(raw_actions, dtype=np.int64)
    values = np.asarray(raw_values, dtype=np.float32).reshape(total_steps, 1)

    return CompactBatch(obs=obs, mask=masks, action=actions, value=values)


def generate_mcts_selfplay_compact_batch(
    num_games: int = 100,
    num_simulations: int = 30,
    start_seed: int = 42,
    temp_steps: int = 12,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> CompactBatch:
    """全速调用底层 Rust 8 线程并行 MCTS 深度推演，秒级产出高质量带 AlphaZero 探索的自博弈样本."""
    raw_obs, raw_masks, raw_actions, raw_values, total_steps = generate_mcts_samples(
        num_games,
        num_simulations,
        start_seed,
        temp_steps,
        dirichlet_alpha,
        dirichlet_eps,
    )

    obs = np.asarray(raw_obs, dtype=np.float32).reshape(total_steps, SplendorDuelEnv.OBS_SIZE)
    masks = np.asarray(raw_masks, dtype=np.uint8).view(bool).reshape(total_steps, SplendorDuelEnv.ACTION_SIZE)
    actions = np.asarray(raw_actions, dtype=np.int64)
    values = np.asarray(raw_values, dtype=np.float32).reshape(total_steps, 1)

    return CompactBatch(obs=obs, mask=masks, action=actions, value=values)


def generate_selfplay_compact_batch(
    net: SplendorNet,
    device: torch.device,
    num_games: int,
    start_seed: int = 42,
    temperature: float = 1.0,
) -> CompactBatch:
    """使用当前神经网络直觉概率进行快速自博弈 (不跑 MCTS，用于超快速粗糙探索)."""
    all_obs: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values: List[float] = []

    net.eval()
    with torch.no_grad():
        for g in range(num_games):
            env = SplendorDuelEnv(seed=start_seed + g)
            obs, info = env.reset()

            raw_trajectory = []
            while not env.is_done:
                acting_player = env.current_player
                mask = info["action_mask"]

                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

                probs, _ = net.predict_action_probs(obs_t, mask_t, temperature=temperature)
                probs_np = probs.cpu().numpy()[0]
                action = int(np.random.choice(len(probs_np), p=probs_np))

                raw_trajectory.append((obs, mask, action, acting_player))
                obs, reward, terminated, truncated, info = env.step(action)
                if truncated:
                    break

            winner = info.get("winner")
            if winner is not None:
                for obs_s, mask_s, act_s, ply_s in raw_trajectory:
                    all_obs.append(obs_s)
                    all_masks.append(mask_s)
                    all_actions.append(act_s)
                    all_values.append(1.0 if ply_s == winner else -1.0)

    total_steps = len(all_actions)
    obs_arr = np.array(all_obs, dtype=np.float32) if total_steps > 0 else np.zeros((0, SplendorDuelEnv.OBS_SIZE), dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=bool) if total_steps > 0 else np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=bool)
    actions_arr = np.array(all_actions, dtype=np.int64)
    values_arr = np.array(all_values, dtype=np.float32).reshape(-1, 1)

    return CompactBatch(obs=obs_arr, mask=masks_arr, action=actions_arr, value=values_arr)


def generate_neural_mcts_selfplay_compact_batch(
    net: SplendorNet,
    device: torch.device,
    num_games: int = 50,
    num_simulations: int = 30,
    start_seed: int = 42,
    temp_threshold_steps: int = 12,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    c_puct: float = 1.5,
) -> CompactBatch:
    """由当前神经网络纯自主引导的 AlphaZero MCTS 自博弈生成器.

    特点:
    - 树搜索先验来自当前神经网络 Policy Head，完全脱离启发式规则打分
    - 前 temp_threshold_steps 步启用温度采样与根节点 Dirichlet 噪声，打破开局死板锁牌模式
    - 后续回合逐渐收敛确定性推演，最终产生胜负鲜明的强化学习轨迹样本
    """
    all_obs: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values: List[float] = []

    mcts = NeuralMCTS(net, device, c_puct=c_puct)
    pbar = Progress(total=num_games, label="Neural-MCTS SelfPlay")

    for g in range(num_games):
        env = SplendorDuelEnv(seed=start_seed + g)
        obs, info = env.reset()

        raw_trajectory = []
        step_count = 0

        while not env.is_done:
            acting_player = env.current_player
            mask = info["action_mask"]

            # 前期探索: 温度 1.0 + 注入 Dirichlet 噪声
            if step_count < temp_threshold_steps:
                temp = 1.0
                add_noise = True
            else:
                temp = 0.1
                add_noise = False

            action, _ = mcts.search(
                env.game,
                num_sims=num_simulations,
                add_dirichlet=add_noise,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                temperature=temp,
            )

            raw_trajectory.append((obs, mask, action, acting_player))
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if truncated:
                break

        winner = info.get("winner")
        if winner is not None:
            for obs_s, mask_s, act_s, ply_s in raw_trajectory:
                all_obs.append(obs_s)
                all_masks.append(mask_s)
                all_actions.append(act_s)
                all_values.append(1.0 if ply_s == winner else -1.0)

        pbar.update(g + 1, extra=f"已累计采样 {len(all_actions)} 步")

    pbar.done(f"完成 {num_games} 局自博弈推演，总样本 {len(all_actions)} 步")

    total_steps = len(all_actions)
    obs_arr = (
        np.array(all_obs, dtype=np.float32)
        if total_steps > 0
        else np.zeros((0, SplendorDuelEnv.OBS_SIZE), dtype=np.float32)
    )
    masks_arr = (
        np.array(all_masks, dtype=bool)
        if total_steps > 0
        else np.zeros((0, SplendorDuelEnv.ACTION_SIZE), dtype=bool)
    )
    actions_arr = np.array(all_actions, dtype=np.int64)
    values_arr = np.array(all_values, dtype=np.float32).reshape(-1, 1)

    return CompactBatch(obs=obs_arr, mask=masks_arr, action=actions_arr, value=values_arr)
