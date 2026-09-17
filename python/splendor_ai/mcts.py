"""Agents and interfaces for Splendor Duel."""

import random
from typing import Optional
import numpy as np
import torch

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


class Agent:
    """智能体统一基类接口."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        raise NotImplementedError


class PolicyNetAgent(Agent):
    """纯神经网络直觉型智能体 (不跑 MCTS 深度推演，毫秒级快速评估)."""

    def __init__(self, net: SplendorNet, device: torch.device, temperature: float = 0.2) -> None:
        self.net = net.to(device)
        self.device = device
        self.temperature = temperature

    def select_action(self, env: SplendorDuelEnv) -> int:
        legals = env.legal_actions
        if len(legals) <= 1:
            return legals[0] if legals else 0

        obs_t = torch.tensor(env.game.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.from_numpy(env.action_mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _, _, _ = self.net.predict_action_probs(obs_t, mask_t, temperature=self.temperature)
            if self.temperature <= 1e-3:
                return int(probs.argmax().item())
            probs_np = probs.cpu().numpy()[0]
            return int(np.random.choice(len(probs_np), p=probs_np))


class HeuristicAgent(Agent):
    """Rust 高性能启发式规则智能体 (算力打分基准)."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        act = env.heuristic_action()
        return act if act is not None else 0


class RandomAgent(Agent):
    """纯随机基准智能体 (探索基线)."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        legals = env.legal_actions
        return random.choice(legals) if legals else 0

