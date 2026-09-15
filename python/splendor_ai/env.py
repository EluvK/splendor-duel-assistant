"""Splendor Duel Gymnasium-style environment."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from splendor_ai._engine import PyGameState


class SplendorDuelEnv:
    """璀璨宝石：对决 (Splendor Duel) 强化学习交互环境.

    遵循标准 Gymnasium 交互协议:
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action_id)
    """

    OBS_SIZE = PyGameState.observation_space_size()  # 1005
    ACTION_SIZE = PyGameState.action_space_size()   # 288

    def __init__(self, seed: Optional[int] = None, max_steps: int = 1500) -> None:
        self.max_steps = max_steps
        self.step_count = 0
        self._seed = seed if seed is not None else 42
        self.game = PyGameState(self._seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境."""
        if seed is not None:
            self._seed = seed
        self.game.reset(self._seed)
        self.step_count = 0

        obs = np.array(self.game.observe(), dtype=np.float32)
        info = self._get_info()
        return obs, info

    def step(
        self, action_id: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行单步动作.

        Args:
            action_id: 0..287 范围内的离散动作 ID.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self.step_count += 1
        acting_player = self.game.current_player()

        raw_obs, terminated, winner = self.game.step(action_id)
        obs = np.array(raw_obs, dtype=np.float32)
        truncated = self.step_count >= self.max_steps and not terminated

        # 胜负回报结算: 从动作执行者视角 (acting_player) 计算回报
        reward = 0.0
        if terminated:
            if winner is not None:
                reward = 1.0 if winner == acting_player else -1.0
            else:
                reward = 0.0

        info = self._get_info()
        info["acting_player"] = acting_player
        info["winner"] = winner

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """提取环境元数据与动作掩码."""
        mask = np.array(self.game.action_mask(), dtype=bool)
        legals = self.game.legal_action_ids()
        scores = self.game.scores()
        crowns = self.game.crowns()

        return {
            "action_mask": mask,
            "legal_actions": legals,
            "current_player": self.game.current_player(),
            "turn_number": self.game.turn_number(),
            "round_number": self.game.round_number(),
            "phase": self.game.phase(),
            "scores": scores,
            "crowns": crowns,
            "step_count": self.step_count,
        }

    @property
    def action_mask(self) -> np.ndarray:
        return np.array(self.game.action_mask(), dtype=bool)

    @property
    def legal_actions(self) -> List[int]:
        return self.game.legal_action_ids()

    @property
    def current_player(self) -> int:
        return self.game.current_player()

    @property
    def turn_number(self) -> int:
        return self.game.turn_number()

    @property
    def round_number(self) -> int:
        return self.game.round_number()

    @property
    def is_done(self) -> bool:
        return self.game.is_done() or self.step_count >= self.max_steps

    @property
    def scores(self) -> Tuple[int, int]:
        """双方当前声望总分 (p0_points, p1_points)."""
        return self.game.scores()

    @property
    def crowns(self) -> Tuple[int, int]:
        """双方当前王冠总数 (p0_crowns, p1_crowns)."""
        return self.game.crowns()

    def heuristic_action(self, seed: Optional[int] = None) -> Optional[int]:
        """获取启发式 AI 推荐的动作 ID."""
        actual_seed = seed if seed is not None else int(np.random.randint(0, 2**31 - 1))
        return self.game.heuristic_action_id(actual_seed)

    def rust_mcts_action(
        self,
        num_sims: int = 50,
        seed: Optional[int] = None,
        max_rollout_steps: int = 15,
    ) -> Optional[int]:
        """获取底层 Rust 原生高性能 MCTS 推荐的动作 ID."""
        return self.game.mcts_action_id(num_sims, seed, max_rollout_steps=max_rollout_steps)

    def clone(self) -> "SplendorDuelEnv":
        """深拷贝环境，用于 MCTS 搜索分支模拟."""
        new_env = SplendorDuelEnv.__new__(SplendorDuelEnv)
        new_env.max_steps = self.max_steps
        new_env.step_count = self.step_count
        new_env._seed = self._seed
        new_env.game = self.game.clone_state()
        return new_env
