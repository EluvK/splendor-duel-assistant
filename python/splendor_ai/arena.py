"""Arena for head-to-head evaluation with paired seeds and seat swapping."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.mcts import Agent
from splendor_ai.progress import Progress


@dataclass
class ArenaResult:
    """竞技场对战结果统计."""

    agent0_name: str
    agent1_name: str
    total_games: int
    agent0_wins: int
    agent1_wins: int
    draws: int
    agent0_win_rate: float
    avg_steps: float
    agent0_as_p0_wins: int
    agent0_as_p1_wins: int


class Arena:
    """成对种子对战竞技场 (消除荷官发牌运气偏差)."""

    def __init__(
        self,
        agent0: Agent,
        agent1: Agent,
        agent0_name: str = "Candidate",
        agent1_name: str = "Baseline",
    ) -> None:
        self.agent0 = agent0
        self.agent1 = agent1
        self.agent0_name = agent0_name
        self.agent1_name = agent1_name

    def play_game(
        self, seed: int, player0: Agent, player1: Agent, max_steps: int = 400
    ) -> Tuple[Optional[int], int]:
        """单局对抗 (返回 winner: 0 或 1, 对局步数)."""
        env = SplendorDuelEnv(seed=seed, max_steps=max_steps)
        env.reset()
        steps = 0

        while not env.is_done:
            steps += 1
            curr_mover = env.current_player
            agent = player0 if curr_mover == 0 else player1
            action = agent.select_action(env)
            env.step(action)

        return env.game.winner(), steps

    def play_match(self, num_pairs: int = 10, base_seed: int = 1000) -> ArenaResult:
        """执行成对种子双向对决.

        每个种子 S 均执行两局:
          - 局 1: Agent 0 (P0) vs Agent 1 (P1)
          - 局 2: Agent 1 (P0) vs Agent 0 (P1) [种子相同]
        总对局数 = 2 * num_pairs
        """
        total_games = num_pairs * 2
        agent0_wins = 0
        agent1_wins = 0
        draws = 0
        agent0_as_p0_wins = 0
        agent0_as_p1_wins = 0
        total_steps = 0

        pbar = Progress(total=total_games, label=f"Arena: {self.agent0_name} vs {self.agent1_name}")

        for i in range(num_pairs):
            seed = base_seed + i * 997

            # 局 1: Agent 0 先手 (P0)
            winner1, steps1 = self.play_game(seed, self.agent0, self.agent1)
            total_steps += steps1
            if winner1 == 0:
                agent0_wins += 1
                agent0_as_p0_wins += 1
            elif winner1 == 1:
                agent1_wins += 1
            else:
                draws += 1
            pbar.update(i * 2 + 1, extra=f"{self.agent0_name} 胜率: {agent0_wins/(i*2+1)*100:.1f}%")

            # 局 2: Agent 1 先手 (P0), Agent 0 后手 (P1)
            winner2, steps2 = self.play_game(seed, self.agent1, self.agent0)
            total_steps += steps2
            if winner2 == 1:
                agent0_wins += 1
                agent0_as_p1_wins += 1
            elif winner2 == 0:
                agent1_wins += 1
            else:
                draws += 1
            pbar.update(i * 2 + 2, extra=f"{self.agent0_name} 胜率: {agent0_wins/(i*2+2)*100:.1f}%")

        pbar.done()

        win_rate = agent0_wins / max(total_games, 1)
        avg_steps = total_steps / max(total_games, 1)

        return ArenaResult(
            agent0_name=self.agent0_name,
            agent1_name=self.agent1_name,
            total_games=total_games,
            agent0_wins=agent0_wins,
            agent1_wins=agent1_wins,
            draws=draws,
            agent0_win_rate=win_rate,
            avg_steps=avg_steps,
            agent0_as_p0_wins=agent0_as_p0_wins,
            agent0_as_p1_wins=agent0_as_p1_wins,
        )
