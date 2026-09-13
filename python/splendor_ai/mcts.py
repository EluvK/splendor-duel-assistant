"""Monte Carlo Tree Search (MCTS) with PUCT and Player-0 Absolute Frame."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


class MCTSNode:
    """MCTS 搜索树节点 (采用 Player-0 绝对坐标价值体系)."""

    def __init__(self, player: int, prior: float = 1.0) -> None:
        self.player = player  # 当前节点轮到谁行动 (0 或 1)
        self.prior = prior    # 来自父节点的先验概率 P(s, a)
        self.n_visits = 0     # 访问次数 N
        self.w_p0 = 0.0       # 累积价值 W (严格在 Player 0 视角, [-1.0, 1.0])
        self.children: Dict[int, MCTSNode] = {}
        self.legal_actions: List[int] = []
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value_p0 = 0.0

    @property
    def q_p0(self) -> float:
        """Player 0 绝对视角的平均价值 Q."""
        if self.n_visits == 0:
            return 0.0
        return self.w_p0 / self.n_visits

    def q_for_mover(self) -> float:
        """站在当前行动玩家视角的期望价值 (行动方为 P0 则是 Q_p0, 为 P1 则是 -Q_p0)."""
        return self.q_p0 if self.player == 0 else -self.q_p0


class MCTS:
    """基于 AlphaZero PUCT 算法的蒙特卡洛树搜索器."""

    def __init__(
        self,
        net: SplendorNet,
        device: torch.device,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ) -> None:
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def search(
        self,
        env: SplendorDuelEnv,
        num_simulations: int = 50,
        add_noise: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """对给定环境状态进行 MCTS 推演.

        Args:
            env: 当前对局环境
            num_simulations: 模拟推演次数
            add_noise: 是否在根节点注入 Dirichlet 探索噪声 (自对弈开启)
            temperature: 最终动作选择的温度系数

        Returns:
            (pi_mcts, selected_action): 搜索策略分布 [256] 和选定动作 ID
        """
        legal_actions = env.legal_actions
        action_size = env.ACTION_SIZE

        # 零开销优化：若合法动作 <= 1，免除一切 MCTS 搜索直接返回
        if len(legal_actions) <= 1:
            pi = np.zeros(action_size, dtype=np.float32)
            act = legal_actions[0] if legal_actions else 0
            pi[act] = 1.0
            return pi, act

        root = MCTSNode(player=env.current_player)
        root.legal_actions = legal_actions

        # 1. 根节点展开与评估
        obs_t = torch.tensor(env.game.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.from_numpy(env.action_mask).unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            probs, value_mover = self.net.predict_action_probs(obs_t, mask_t, temperature=1.0)
            priors = probs.cpu().numpy()[0]
            val_mover = float(value_mover.item())

        # 将网络给出的行动方价值转换为 Player-0 绝对价值
        v_p0 = val_mover if root.player == 0 else -val_mover
        root.w_p0 = v_p0
        root.n_visits = 1

        # 注入 Dirichlet 探索噪声
        priors_legal = np.array([priors[a] for a in legal_actions], dtype=np.float32)
        priors_legal = priors_legal / max(priors_legal.sum(), 1e-8)

        if add_noise and len(legal_actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            priors_legal = (1 - self.dirichlet_eps) * priors_legal + self.dirichlet_eps * noise

        for i, a in enumerate(legal_actions):
            root.children[a] = MCTSNode(player=root.player, prior=float(priors_legal[i]))
        root.is_expanded = True

        # 2. 执行 N 次推演模拟
        for _ in range(num_simulations):
            sim_env = env.clone()
            node = root
            search_path = [node]

            # (A) Selection: 沿着树向下选择 PUCT 最大子节点
            while node.is_expanded and not node.is_terminal:
                action, next_node = self._select_child(node)
                sim_env.step(action)
                node = next_node
                search_path.append(node)

            # (B) Evaluation & Expansion
            if sim_env.is_done:
                node.is_terminal = True
                winner = sim_env.game.winner()
                if winner == 0:
                    leaf_v_p0 = 1.0
                elif winner == 1:
                    leaf_v_p0 = -1.0
                else:
                    leaf_v_p0 = 0.0
                node.terminal_value_p0 = leaf_v_p0
            else:
                node.player = sim_env.current_player
                node.legal_actions = sim_env.legal_actions

                s_obs_t = torch.tensor(sim_env.game.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
                s_mask_t = torch.from_numpy(sim_env.action_mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    leaf_probs, leaf_val_mover = self.net.predict_action_probs(s_obs_t, s_mask_t)
                    leaf_priors = leaf_probs.cpu().numpy()[0]
                    val_mover = float(leaf_val_mover.item())

                leaf_v_p0 = val_mover if node.player == 0 else -val_mover

                for a in node.legal_actions:
                    node.children[a] = MCTSNode(player=node.player, prior=float(leaf_priors[a]))
                node.is_expanded = True

            # (C) Backup: 沿着访问路径将 Player-0 绝对价值反向回传
            for path_node in search_path:
                path_node.n_visits += 1
                path_node.w_p0 += leaf_v_p0

        # 3. 提取最终策略分布
        counts = np.zeros(action_size, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.n_visits

        if temperature <= 1e-3:
            # 贪心选择最高访问量动作
            best_a = int(counts.argmax())
            pi = np.zeros(action_size, dtype=np.float32)
            pi[best_a] = 1.0
            return pi, best_a
        else:
            # 带温度采样
            counts_temp = counts ** (1.0 / temperature)
            total = counts_temp.sum()
            pi = counts_temp / (total if total > 0 else 1.0)
            selected_a = int(np.random.choice(action_size, p=pi))
            return pi, selected_a

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """依据 PUCT 公式在当前节点选择最值得探索的子节点."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        total_sqrt = math.sqrt(max(node.n_visits, 1))

        for action, child in node.children.items():
            # Q 站在父节点行动方的视角 (避免交替负号错误)
            q_val = child.q_for_mover()
            u_val = self.c_puct * child.prior * (total_sqrt / (1.0 + child.n_visits))
            score = q_val + u_val

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


class Agent:
    """智能体统一基类接口."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        raise NotImplementedError


class MCTSAgent(Agent):
    """带 MCTS 深度推演的神经网络智能体."""

    def __init__(
        self,
        net: SplendorNet,
        device: torch.device,
        num_sims: int = 50,
        c_puct: float = 1.5,
        temperature: float = 0.0,
    ) -> None:
        self.mcts = MCTS(net, device, c_puct=c_puct)
        self.num_sims = num_sims
        self.temperature = temperature

    def select_action(self, env: SplendorDuelEnv) -> int:
        _, action = self.mcts.search(
            env, num_simulations=self.num_sims, add_noise=False, temperature=self.temperature
        )
        return action


class PolicyNetAgent(Agent):
    """纯网络直觉型智能体 (不跑 MCTS 搜索, argmax)."""

    def __init__(self, net: SplendorNet, device: torch.device) -> None:
        self.net = net.to(device)
        self.device = device

    def select_action(self, env: SplendorDuelEnv) -> int:
        legals = env.legal_actions
        if len(legals) <= 1:
            return legals[0] if legals else 0

        obs_t = torch.tensor(env.game.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.from_numpy(env.action_mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.net.predict_action_probs(obs_t, mask_t, temperature=0.0)
            return int(probs.argmax().item())


class HeuristicAgent(Agent):
    """Rust 高性能启发式规则智能体."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        act = env.heuristic_action()
        return act if act is not None else 0


class RustMCTSAgent(Agent):
    """纯 Rust 底层原生高性能 MCTS 智能体 (微秒级推演，棋力极强)."""

    def __init__(self, num_sims: int = 50) -> None:
        self.num_sims = num_sims

    def select_action(self, env: SplendorDuelEnv) -> int:
        act = env.rust_mcts_action(num_sims=self.num_sims)
        return act if act is not None else 0


class RandomAgent(Agent):
    """纯随机基准智能体."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        import random
        legals = env.legal_actions
        return random.choice(legals) if legals else 0
