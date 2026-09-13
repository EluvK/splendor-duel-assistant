"""Self-play and heuristic dataset generators for Splendor Duel."""

from typing import List, Tuple
import numpy as np
import torch

from splendor_ai.dataset import Sample
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


def generate_heuristic_game(seed: int) -> List[Sample]:
    """使用高性能启发式 AI 完整运行一局，并提取状态-策略-胜负样本."""
    env = SplendorDuelEnv(seed=seed)
    obs, info = env.reset()

    # 暂存单局轨迹: (obs, mask, action_id, acting_player)
    raw_trajectory: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    step = 0

    while not env.is_done:
        step += 1
        acting_player = env.current_player
        mask = info["action_mask"]

        action = env.heuristic_action(seed=seed + step * 31)
        if action is None:
            break

        raw_trajectory.append((obs, mask, action, acting_player))
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            break

    winner = info.get("winner")
    if winner is None:
        return []

    # 终局回溯：根据终局胜负给每一步赋值回报
    samples: List[Sample] = []
    for obs_step, mask_step, action_step, player_step in raw_trajectory:
        target_policy = np.zeros(SplendorDuelEnv.ACTION_SIZE, dtype=np.float32)
        target_policy[action_step] = 1.0

        # 从该步行动方视角计算回报
        target_value = 1.0 if player_step == winner else -1.0

        samples.append(
            Sample(
                obs=obs_step,
                mask=mask_step,
                target_policy=target_policy,
                target_value=target_value,
            )
        )

    return samples


def generate_heuristic_dataset(
    num_games: int, start_seed: int = 0
) -> List[Sample]:
    """批量生成启发式专家轨迹数据集."""
    all_samples: List[Sample] = []
    for g in range(num_games):
        seed = start_seed + g
        game_samples = generate_heuristic_game(seed)
        all_samples.extend(game_samples)
    return all_samples


def generate_selfplay_game(
    net: SplendorNet,
    device: torch.device,
    seed: int,
    temperature: float = 1.0,
) -> List[Sample]:
    """使用当前神经网络策略模型进行自博弈对弈，并提取样本."""
    env = SplendorDuelEnv(seed=seed)
    obs, info = env.reset()

    # (obs, mask, policy_probs, acting_player)
    raw_trajectory: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    net.eval()

    with torch.no_grad():
        while not env.is_done:
            acting_player = env.current_player
            mask = info["action_mask"]

            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

            probs, _ = net.predict_action_probs(obs_t, mask_t, temperature=temperature)
            probs_np = probs.cpu().numpy()[0]

            # 根据概率分布采样合法动作
            action = int(np.random.choice(len(probs_np), p=probs_np))

            raw_trajectory.append((obs, mask, probs_np, acting_player))
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                break

    winner = info.get("winner")
    if winner is None:
        return []

    samples: List[Sample] = []
    for obs_step, mask_step, probs_step, player_step in raw_trajectory:
        target_value = 1.0 if player_step == winner else -1.0
        samples.append(
            Sample(
                obs=obs_step,
                mask=mask_step,
                target_policy=probs_step,
                target_value=target_value,
            )
        )

    return samples


def generate_selfplay_dataset(
    net: SplendorNet,
    device: torch.device,
    num_games: int,
    start_seed: int = 0,
    temperature: float = 1.0,
) -> List[Sample]:
    """批量生成神经网络自博弈对弈数据集."""
    all_samples: List[Sample] = []
    for g in range(num_games):
        seed = start_seed + g
        game_samples = generate_selfplay_game(net, device, seed, temperature=temperature)
        all_samples.extend(game_samples)
    return all_samples
