"""Arena for head-to-head evaluation with paired seeds and seat swapping."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple
import time

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.mcts import Agent
from splendor_ai.progress import Progress


@dataclass
class SingleGameResult:
    """单局对战明细数据."""

    winner: Optional[int]
    steps: int
    turns: int
    rounds: int
    p0_score: int
    p1_score: int
    p0_crowns: int
    p1_crowns: int
    reason: str


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
    avg_win_steps: float = 0.0
    avg_lose_steps: float = 0.0
    avg_turns: float = 0.0
    avg_rounds: float = 0.0
    avg_win_rounds: float = 0.0
    avg_lose_rounds: float = 0.0
    p0_seat_win_rate: float = 0.0
    p1_seat_win_rate: float = 0.0
    reasons: Optional[Dict[str, int]] = None


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
    ) -> SingleGameResult:
        """单局对抗 (返回单局详细对弈指标)."""
        env = SplendorDuelEnv(seed=seed, max_steps=max_steps)
        env.reset()
        steps = 0

        while not env.is_done:
            steps += 1
            curr_mover = env.current_player
            agent = player0 if curr_mover == 0 else player1
            action = agent.select_action(env)
            env.step(action)

        winner = env.game.winner()
        p0_score, p1_score = env.scores
        p0_crowns, p1_crowns = env.crowns
        turns = env.game.turn_number()
        rounds = env.round_number

        # 胜负判定原因推断
        reason = "draw"
        if winner is not None:
            win_score = p0_score if winner == 0 else p1_score
            win_crowns = p0_crowns if winner == 0 else p1_crowns
            if win_score >= 20:
                reason = "20_points"
            elif win_crowns >= 10:
                reason = "10_crowns"
            else:
                reason = "10_color_points"

        return SingleGameResult(
            winner=winner,
            steps=steps,
            turns=turns,
            rounds=rounds,
            p0_score=p0_score,
            p1_score=p1_score,
            p0_crowns=p0_crowns,
            p1_crowns=p1_crowns,
            reason=reason,
        )

    def play_match(
        self,
        num_pairs: int = 10,
        base_seed: int = 1000,
        workers: int = 0,
    ) -> ArenaResult:
        """执行成对种子双向对决 (支持全核高并发评估).

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
        total_turns = 0
        total_rounds = 0

        p0_seat_total_wins = 0
        agent0_win_steps: List[int] = []
        agent0_lose_steps: List[int] = []
        agent0_win_rounds: List[int] = []
        agent0_lose_rounds: List[int] = []
        reasons_count: Dict[str, int] = {
            "20_points": 0,
            "10_crowns": 0,
            "10_color_points": 0,
            "draw": 0,
        }

        actual_workers = min(8, os.cpu_count() or 4) if workers <= 0 else workers
        actual_workers = min(actual_workers, total_games)
        pbar = Progress(total=total_games, label=f"Arena: {self.agent0_name} vs {self.agent1_name}")

        # 构建所有成对任务: (pair_idx, is_swap, seed, p0_agent, p1_agent)
        tasks = []
        for i in range(num_pairs):
            seed = base_seed + i * 997
            tasks.append((i, False, seed, self.agent0, self.agent1))
            tasks.append((i, True, seed, self.agent1, self.agent0))

        def _record_result(res: SingleGameResult, is_swap: bool, current_done: int):
            nonlocal agent0_wins, agent1_wins, draws, agent0_as_p0_wins, agent0_as_p1_wins
            nonlocal total_steps, total_turns, total_rounds, p0_seat_total_wins

            total_steps += res.steps
            total_turns += res.turns
            total_rounds += res.rounds
            reasons_count[res.reason] = reasons_count.get(res.reason, 0) + 1

            if not is_swap:
                # 局 1: Agent 0 是 P0
                if res.winner == 0:
                    agent0_wins += 1
                    agent0_as_p0_wins += 1
                    p0_seat_total_wins += 1
                    agent0_win_steps.append(res.steps)
                    agent0_win_rounds.append(res.rounds)
                elif res.winner == 1:
                    agent1_wins += 1
                    agent0_lose_steps.append(res.steps)
                    agent0_lose_rounds.append(res.rounds)
                else:
                    draws += 1
            else:
                # 局 2: Agent 1 是 P0, Agent 0 是 P1
                if res.winner == 1:
                    agent0_wins += 1
                    agent0_as_p1_wins += 1
                    agent0_win_steps.append(res.steps)
                    agent0_win_rounds.append(res.rounds)
                elif res.winner == 0:
                    agent1_wins += 1
                    p0_seat_total_wins += 1
                    agent0_lose_steps.append(res.steps)
                    agent0_lose_rounds.append(res.rounds)
                else:
                    draws += 1

            pbar.update(current_done, extra=f"{self.agent0_name} 胜率: {agent0_wins/current_done*100:.1f}%")

        if actual_workers > 1:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_meta = {
                    executor.submit(self.play_game, seed, p0, p1): is_swap
                    for (_, is_swap, seed, p0, p1) in tasks
                }
                done_count = 0
                for future in as_completed(future_to_meta):
                    is_swap = future_to_meta[future]
                    res = future.result()
                    done_count += 1
                    _record_result(res, is_swap, done_count)
        else:
            for idx, (_, is_swap, seed, p0, p1) in enumerate(tasks):
                res = self.play_game(seed, p0, p1)
                _record_result(res, is_swap, idx + 1)

        pbar.done()

        win_rate = agent0_wins / max(total_games, 1)
        avg_steps = total_steps / max(total_games, 1)
        avg_turns = total_turns / max(total_games, 1)
        avg_rounds = total_rounds / max(total_games, 1)
        avg_win_steps = float(sum(agent0_win_steps) / len(agent0_win_steps)) if agent0_win_steps else 0.0
        avg_lose_steps = float(sum(agent0_lose_steps) / len(agent0_lose_steps)) if agent0_lose_steps else 0.0
        avg_win_rounds = float(sum(agent0_win_rounds) / len(agent0_win_rounds)) if agent0_win_rounds else 0.0
        avg_lose_rounds = float(sum(agent0_lose_rounds) / len(agent0_lose_rounds)) if agent0_lose_rounds else 0.0
        p0_seat_win_rate = p0_seat_total_wins / max(total_games, 1)
        p1_seat_win_rate = (total_games - p0_seat_total_wins - draws) / max(total_games, 1)

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
            avg_win_steps=avg_win_steps,
            avg_lose_steps=avg_lose_steps,
            avg_turns=avg_turns,
            avg_rounds=avg_rounds,
            avg_win_rounds=avg_win_rounds,
            avg_lose_rounds=avg_lose_rounds,
            p0_seat_win_rate=p0_seat_win_rate,
            p1_seat_win_rate=p1_seat_win_rate,
            reasons=reasons_count,
        )
