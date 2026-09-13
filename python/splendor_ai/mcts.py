"""Agents and high-performance MCTS interfaces for Splendor Duel."""

from typing import List, Optional
import numpy as np
import torch

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


"""Agents and high-performance MCTS interfaces for Splendor Duel."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


class Agent:
    """智能体统一基类接口."""

    def select_action(self, env: SplendorDuelEnv) -> int:
        raise NotImplementedError


class MCTSNode:
    """AlphaZero MCTS 树节点."""

    __slots__ = (
        "state",
        "parent",
        "action_from_parent",
        "prior",
        "visits",
        "value_sum_p0",
        "children",
        "is_expanded",
        "is_terminal",
        "player",
        "winner",
    )

    def __init__(
        self,
        state: Optional[any] = None,
        parent: Optional["MCTSNode"] = None,
        action_from_parent: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = prior
        self.visits = 0
        self.value_sum_p0 = 0.0  # Player 0 绝对累积价值
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded = False
        if state is not None:
            self.is_terminal = state.is_done()
            self.player = state.current_player()
            self.winner = state.winner()
        else:
            self.is_terminal = False
            self.player = 0
            self.winner = None


class NeuralMCTS:
    """纯神经网络驱动的 AlphaZero MCTS 搜索器.

    特点:
    - 树先验来自 Policy Head 输出 (脱离手工启发式规则)
    - 叶子节点价值直接由 Value Head 预测 (无手工 Rollout 偏见)
    - 支持根节点 Dirichlet 噪声注入 (破除开局盲区)
    - 支持自博弈温度轮盘赌采样 (产出平滑 MCTS 访问分布 pi)
    """

    def __init__(self, net: SplendorNet, device: torch.device, c_puct: float = 1.5) -> None:
        self.net = net.to(device)
        self.device = device
        self.c_puct = c_puct

    def search(
        self,
        game_state: any,
        num_sims: int = 40,
        add_dirichlet: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        temperature: float = 1.0,
    ) -> Tuple[int, np.ndarray]:
        """执行 MCTS 搜索并返回 (选取的动作, MCTS 访问频率分布 pi)."""
        legals = game_state.legal_action_ids()
        pi = np.zeros(SplendorDuelEnv.ACTION_SIZE, dtype=np.float32)

        if not legals:
            return 0, pi

        # 唯一动作快捷路径
        if len(legals) == 1:
            pi[legals[0]] = 1.0
            return legals[0], pi

        # 初始化根节点
        root = MCTSNode(state=game_state.clone_state())
        self.net.eval()

        # 展开根节点先验
        obs_t = torch.tensor(root.state.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(root.state.action_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs_t, val_t = self.net.predict_action_probs(obs_t, mask_t, temperature=1.0)
        priors = probs_t[0].cpu().numpy()

        # 注入 Dirichlet 探索噪声 (自博弈关键)
        if add_dirichlet and len(legals) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legals))
            for idx, act in enumerate(legals):
                priors[act] = (1.0 - dirichlet_eps) * priors[act] + dirichlet_eps * noise[idx]
            # 重新在合法集合上归一化
            legal_sum = priors[legals].sum()
            if legal_sum > 1e-6:
                priors[legals] /= legal_sum

        for act in legals:
            root.children[act] = MCTSNode(
                state=None, parent=root, action_from_parent=act, prior=float(priors[act])
            )
        root.is_expanded = True

        # 开始树搜索推演
        for _ in range(num_sims):
            node = root

            # 1. Selection
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)

            # 2. Expansion & Evaluation
            if node.is_terminal:
                if node.winner == 0:
                    v_p0 = 1.0
                elif node.winner == 1:
                    v_p0 = -1.0
                else:
                    v_p0 = 0.0
            else:
                obs_t = torch.tensor(node.state.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.tensor(node.state.action_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    probs_t, val_t = self.net.predict_action_probs(obs_t, mask_t, temperature=1.0)
                child_priors = probs_t[0].cpu().numpy()
                v_curr = val_t[0, 0].item()
                v_p0 = v_curr if node.player == 0 else -v_curr

                child_legals = node.state.legal_action_ids()
                for act in child_legals:
                    node.children[act] = MCTSNode(
                        state=None, parent=node, action_from_parent=act, prior=float(child_priors[act])
                    )
                node.is_expanded = True

            # 3. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum_p0 += v_p0
                curr = curr.parent

        # 汇总根节点访问频次生成训练目标 pi
        for act, child in root.children.items():
            pi[act] = child.visits
        sum_visits = pi.sum()
        if sum_visits > 0:
            pi /= sum_visits

        # 根据温度选取行动
        if temperature <= 1e-2:
            action = int(np.argmax(pi))
        else:
            # 引入温度缩放
            v_temp = pi ** (1.0 / max(temperature, 1e-3))
            s_temp = v_temp.sum()
            probs_sample = v_temp / s_temp if s_temp > 1e-6 else pi
            action = int(np.random.choice(len(probs_sample), p=probs_sample))

        return action, pi

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        total_sqrt = math.sqrt(node.visits) if node.visits > 0 else 1.0
        best_score = -float("inf")
        best_child = None

        for act, child in node.children.items():
            q_p0 = child.value_sum_p0 / child.visits if child.visits > 0 else 0.0
            q_mover = q_p0 if node.player == 0 else -q_p0
            u = self.c_puct * child.prior * (total_sqrt / (1.0 + child.visits))
            score = q_mover + u

            if score > best_score:
                best_score = score
                best_child = child

        # 延迟状态懒克隆 (按需 step)
        if best_child.state is None:
            child_state = node.state.clone_state()
            child_state.step(best_child.action_from_parent)
            best_child.state = child_state
            best_child.is_terminal = child_state.is_done()
            best_child.winner = child_state.winner()
            best_child.player = child_state.current_player()

        return best_child


class NeuralMCTSAgent(Agent):
    """纯神经网络驱动的高性能 MCTS 智能体 (AlphaZero 原生架构)."""

    def __init__(
        self,
        net: SplendorNet,
        device: torch.device,
        num_sims: int = 40,
        c_puct: float = 1.5,
        temperature: float = 0.0,
    ) -> None:
        self.mcts = NeuralMCTS(net, device, c_puct=c_puct)
        self.num_sims = num_sims
        self.temperature = temperature

    def select_action(self, env: SplendorDuelEnv) -> int:
        action, _ = self.mcts.search(env.game, num_sims=self.num_sims, temperature=self.temperature)
        return action


class MCTSAgent(Agent):
    """纯 Rust 底层原生高性能 MCTS 智能体 (单步微秒级推演，棋力超越启发式规则)."""

    def __init__(self, num_sims: int = 50) -> None:
        self.num_sims = num_sims

    def select_action(self, env: SplendorDuelEnv) -> int:
        act = env.rust_mcts_action(num_sims=self.num_sims)
        return act if act is not None else 0


# 保持命名兼容
RustMCTSAgent = MCTSAgent


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
            probs, _ = self.net.predict_action_probs(obs_t, mask_t, temperature=self.temperature)
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
        import random
        legals = env.legal_actions
        return random.choice(legals) if legals else 0
