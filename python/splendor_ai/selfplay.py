"""Self-play and heuristic dataset generators for Splendor Duel."""

from typing import List, Tuple
import numpy as np
import torch

from splendor_ai._engine import generate_heuristic_samples
from splendor_ai.dataset import CompactBatch
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


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


def generate_selfplay_compact_batch(
    net: SplendorNet,
    device: torch.device,
    num_games: int,
    start_seed: int = 42,
    temperature: float = 1.0,
) -> CompactBatch:
    """使用当前神经网络策略模型进行自博弈对弈，并提取紧凑张量样本."""
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

                # 依据策略概率分布采样动作
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
    obs_arr = np.array(all_obs, dtype=np.float32) if total_steps > 0 else np.zeros((0, 725), dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=bool) if total_steps > 0 else np.zeros((0, 256), dtype=bool)
    actions_arr = np.array(all_actions, dtype=np.int64)
    values_arr = np.array(all_values, dtype=np.float32).reshape(-1, 1)

    return CompactBatch(obs=obs_arr, mask=masks_arr, action=actions_arr, value=values_arr)
