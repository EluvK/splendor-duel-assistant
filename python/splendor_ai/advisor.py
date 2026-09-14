"""Training diagnostics, early stopping, and adaptive hyperparameter advisor for Splendor Duel self-play."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class HealthStatus(str, Enum):
    HEALTHY = "正常健康 (Healthy)"
    STAGNANT = "停滞预警 (Stagnant)"
    COLLAPSED = "模式塌陷/回音室 (Mode Collapse / Echo Chamber)"
    DEGRADED = "严重退化 (Degraded)"
    CYCLING = "胜率震荡 (Cycling)"
    ANOMALOUS = "对局异常 (Pace Anomaly)"
    EXPLODED = "梯度发散/直觉崩溃 (Diverged)"
    BASELINE_FAULT = "基准断层异动 (Baseline Fault)"


class MetricLevel(str, Enum):
    ABNORMAL_LOW = "异常偏低"    # 🚨
    WARNING_LOW = "偏低预警"     # ⚠️
    HEALTHY = "绝对健康"        # 🟢
    WARNING_HIGH = "偏高预警"    # ⚠️
    ABNORMAL_HIGH = "异常偏高"   # 🚨


@dataclass
class MetricEval:
    """单一关键维度的健康状态评测结果."""

    value: float
    level: MetricLevel
    emoji: str
    target_range: str
    tag: str
    comment: str


@dataclass
class IterationRecord:
    """自博弈单轮迭代全量训练与评测记录."""

    iteration: int
    train_loss: float
    policy_loss: float
    value_loss: float
    top1_acc: float
    top3_acc: float
    win_rate: float
    promoted: bool
    candidate_wins: int
    baseline_wins: int
    draws: int
    reasons: Dict[str, int]
    avg_rounds: float
    avg_steps: float
    lr: float
    samples_added: int
    buffer_size: int
    current_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationDecision:
    """自博弈及时终止判定结果."""

    should_terminate: bool
    reason_type: str
    description: str
    metric_evidence: str


@dataclass
class AdviceReport:
    """自适应训练调优与下一阶段建议看板."""

    health_status: HealthStatus
    summary: str
    root_causes: List[str]
    recommended_params: Dict[str, Any]
    recommended_command: str


class TrainingAdvisor:
    """自博弈智能健康监测、全场景异常及时终止与自适应参数建议器."""

    def __init__(
        self,
        stagnation_patience: int = 15,
        collapse_window: int = 6,
        collapse_top1_thresh: float = 0.975,
        collapse_entropy_thresh: float = 0.15,
        degradation_patience: int = 3,
        degradation_winrate_thresh: float = 0.20,
        echo_chamber_patience: int = 4,
        divergence_patience: int = 3,
    ) -> None:
        self.stagnation_patience = stagnation_patience
        self.collapse_window = collapse_window
        self.collapse_top1_thresh = collapse_top1_thresh
        self.collapse_entropy_thresh = collapse_entropy_thresh
        self.degradation_patience = degradation_patience
        self.degradation_winrate_thresh = degradation_winrate_thresh
        self.echo_chamber_patience = echo_chamber_patience
        self.divergence_patience = divergence_patience
        self.anomalous_patience = 3
        self.cycling_patience = 6

        self.history: List[IterationRecord] = []
        self.consecutive_failures: int = 0
        self.high_confidence_streak: int = 0
        self.echo_chamber_streak: int = 0
        self.divergence_streak: int = 0
        self.degraded_streak: int = 0
        self.anomalous_streak: int = 0
        self.cycling_streak: int = 0
        self.fault_streak: int = 0

    @staticmethod
    def evaluate_loss(loss: float) -> MetricEval:
        """评估训练总 Loss (CrossEntropy + 1.0 * MSE).

        - < 0.60: 🚨 异常偏低（丧失探索，网络陷入回音室）
        - 0.60 ~ 0.90: ⚠️ 偏低预警（拟合过深或多样性降低）
        - 0.90 ~ 1.35: 🟢 绝对健康运行区间（前沿棋力健康拟合）
        - 1.35 ~ 2.00: ⚠️ 偏高预警（欠拟合或处于学习初期）
        - > 2.00: 🚨 异常偏高（梯度爆炸、学习率过大震荡）
        """
        if loss < 0.60:
            return MetricEval(
                value=loss,
                level=MetricLevel.ABNORMAL_LOW,
                emoji="🚨",
                target_range="0.90~1.35",
                tag="异常偏低(回音室)",
                comment="网络陷入回音室，对局同质化与过度拟合",
            )
        if loss < 0.90:
            return MetricEval(
                value=loss,
                level=MetricLevel.WARNING_LOW,
                emoji="⚠️",
                target_range="0.90~1.35",
                tag="偏低预警",
                comment="拟合略深，需关注探索空间",
            )
        if loss <= 1.35:
            return MetricEval(
                value=loss,
                level=MetricLevel.HEALTHY,
                emoji="🟢",
                target_range="0.90~1.35",
                tag="健康运行",
                comment="前沿棋力健康拟合区间",
            )
        if loss <= 2.00:
            return MetricEval(
                value=loss,
                level=MetricLevel.WARNING_HIGH,
                emoji="⚠️",
                target_range="0.90~1.35",
                tag="偏高预警",
                comment="欠拟合或处于自博弈初期/样本扰动",
            )
        return MetricEval(
            value=loss,
            level=MetricLevel.ABNORMAL_HIGH,
            emoji="🚨",
            target_range="0.90~1.35",
            tag="异常偏高(发散)",
            comment="梯度爆炸、学习率过大或特征破坏",
        )

    @staticmethod
    def evaluate_top1(top1: float) -> MetricEval:
        """评估 Top-1 先验预测准确率 (网络 vs MCTS 搜索目标分布).

        - < 70.0%: 🚨 异常偏低（网络直觉崩溃、缺乏主见）
        - 70.0% ~ 82.0%: ⚠️ 偏低预警（策略直觉偏弱，MCTS 搜索负担过重）
        - 82.0% ~ 89.0%: 🟢 绝对健康运行区间（常规步敏捷，关键步有深算空间）
        - 89.0% ~ 93.0%: ⚠️ 偏高预警（策略过度自信边缘，留意探索多样性）
        - > 93.0%: 🚨 异常偏高（过拟合、丧失策略改进增量）
        """
        pct = top1 * 100
        if top1 < 0.70:
            return MetricEval(
                value=pct,
                level=MetricLevel.ABNORMAL_LOW,
                emoji="🚨",
                target_range="82.0%~89.0%",
                tag="异常偏低(直觉弱)",
                comment="网络直觉崩溃、缺乏主见，MCTS 纠偏负担过重",
            )
        if top1 < 0.82:
            return MetricEval(
                value=pct,
                level=MetricLevel.WARNING_LOW,
                emoji="⚠️",
                target_range="82.0%~89.0%",
                tag="偏低预警",
                comment="直觉偏弱，常规步命中率待提升",
            )
        if top1 <= 0.89:
            return MetricEval(
                value=pct,
                level=MetricLevel.HEALTHY,
                emoji="🟢",
                target_range="82.0%~89.0%",
                tag="健康运行",
                comment="黄金平衡：常规步敏捷，关键步有深算纠偏空间",
            )
        if top1 <= 0.93:
            return MetricEval(
                value=pct,
                level=MetricLevel.WARNING_HIGH,
                emoji="⚠️",
                target_range="82.0%~89.0%",
                tag="偏高预警",
                comment="策略过度自信边缘，需保持探索噪声",
            )
        return MetricEval(
            value=pct,
            level=MetricLevel.ABNORMAL_HIGH,
            emoji="🚨",
            target_range="82.0%~89.0%",
            tag="异常偏高(过拟合)",
            comment="过拟合、先验压制 MCTS，丧失策略改进增量",
        )

    @staticmethod
    def evaluate_winrate(win_rate: float) -> MetricEval:
        """评估候选模型对基准主力的对抗胜率.

        - < 42.0%: 🚨 异常偏低（大幅负向退步）
        - 42.0% ~ 52.0%: ⚠️ 偏低预警（微弱落后，未能超越基准门禁）
        - 52.0% ~ 58.0%: 🟢 绝对健康运行区间（教科书级平稳迭代进化）
        - 58.0% ~ 75.0%: 🟢 强势进化（突破性增益）
        - > 75.0%: 🚨 异常偏高（两代突然断层，通常说明老模型某处坏了或评估偏差）
        """
        pct = win_rate * 100
        if win_rate < 0.42:
            return MetricEval(
                value=pct,
                level=MetricLevel.ABNORMAL_LOW,
                emoji="🚨",
                target_range="52.0%~58.0%",
                tag="异常偏低(负退化)",
                comment="大幅负向退步，单轮参数更新破坏已有棋力",
            )
        if win_rate < 0.52:
            return MetricEval(
                value=pct,
                level=MetricLevel.WARNING_LOW,
                emoji="⚠️",
                target_range="52.0%~58.0%",
                tag="偏低预警",
                comment="微弱落后或势均力敌，未突破门禁",
            )
        if win_rate <= 0.58:
            return MetricEval(
                value=pct,
                level=MetricLevel.HEALTHY,
                emoji="🟢",
                target_range="52.0%~58.0%",
                tag="健康进化",
                comment="教科书级平稳迭代进化区间",
            )
        if win_rate <= 0.75:
            return MetricEval(
                value=pct,
                level=MetricLevel.HEALTHY,
                emoji="🟢",
                target_range="52.0%~58.0%",
                tag="强势突破",
                comment="候选模型展现出明显的进攻与破局优势",
            )
        return MetricEval(
            value=pct,
            level=MetricLevel.ABNORMAL_HIGH,
            emoji="🚨",
            target_range="52.0%~58.0%",
            tag="异常偏高(断层异动)",
            comment="两代突然断层，通常说明老模型陷入死锁漏洞或评估偏倚",
        )

    @staticmethod
    def compute_shannon_diversity(reasons: Dict[str, int]) -> float:
        """计算三大胜因 (20声望 / 10皇冠 / 10单色) 的香农多样性指数 (Shannon Entropy).

        若胜因单一，熵值为 0.0；若三者均等 (各 1/3)，最大熵约为 1.0986。
        """
        keys = ["20_points", "10_crowns", "10_color_points"]
        counts = [max(0, reasons.get(k, 0)) for k in keys]
        total = sum(counts)
        if total <= 0:
            return 0.0

        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log(p)
        return entropy

    def step(
        self, record: IterationRecord
    ) -> Tuple[HealthStatus, Optional[TerminationDecision], Optional[AdviceReport]]:
        """录入一轮迭代数据，进行全场景健康分析与早停裁决."""
        self.history.append(record)

        if record.promoted:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        entropy = self.compute_shannon_diversity(record.reasons)

        # 1. 传统超高 Top-1 + 单一胜因塌陷
        if record.top1_acc >= self.collapse_top1_thresh and entropy <= self.collapse_entropy_thresh:
            self.high_confidence_streak += 1
        else:
            self.high_confidence_streak = 0

        # 2. 回音室效应监测 (Loss < 0.60 且 Top-1 > 0.93，网络陷入自娱自乐与同质拟合)
        if record.train_loss < 0.60 and record.top1_acc > 0.93:
            self.echo_chamber_streak += 1
        else:
            self.echo_chamber_streak = 0

        # 3. 梯度发散/直觉崩溃监测 (Loss > 2.00 或 Top-1 < 0.70)
        if record.train_loss > 2.00 or record.top1_acc < 0.70:
            self.divergence_streak += 1
        else:
            self.divergence_streak = 0

        # 4. 胜率暴跌与严重退化监测
        if record.win_rate <= self.degradation_winrate_thresh:
            self.degraded_streak += 1
        else:
            self.degraded_streak = 0

        # 5. 两代异常断层监测 (胜率 > 75%)
        if record.win_rate > 0.75:
            self.fault_streak += 1
        else:
            self.fault_streak = 0

        # 6. 对局节奏异常判断 (单局走子严重拖局/死循环或异常猝死)
        is_anomalous_pace = record.avg_rounds > 85.0 or (record.avg_rounds > 0 and record.avg_rounds < 35.0)
        if is_anomalous_pace:
            self.anomalous_streak += 1
        else:
            self.anomalous_streak = 0

        # 7. 胜率循环震荡判断 (近 6 轮存在极差超 45% 的两极分化)
        is_cycling = False
        if len(self.history) >= 6:
            recent_rates = [r.win_rate for r in self.history[-6:]]
            spread = max(recent_rates) - min(recent_rates)
            if spread >= 0.449 and (max(recent_rates) >= 0.60 and min(recent_rates) <= 0.25):
                is_cycling = True

        if is_cycling:
            self.cycling_streak += 1
        else:
            self.cycling_streak = 0

        status = self._evaluate_health(record, entropy)
        decision = self._check_termination(record, entropy, status)
        advice = None
        if decision and decision.should_terminate:
            advice = self.generate_advice(record, status, decision)

        return status, decision, advice

    def _evaluate_health(self, record: IterationRecord, entropy: float) -> HealthStatus:
        """评估当前轮次的实时健康度状态 (按紧急与严重程度从高到低识别)."""
        # 1. 梯度发散与直觉崩溃 (Loss > 2.0 或 Top-1 < 70% 持续)
        if self.divergence_streak >= self.divergence_patience:
            return HealthStatus.EXPLODED

        # 2. 灾难性退化最优先暴露 (胜率暴跌至 20% 以下)
        if self.degraded_streak >= self.degradation_patience:
            return HealthStatus.DEGRADED

        # 3. 严重模式塌陷与回音室 (Top-1 畸高/胜因单一 或 Loss 极低锁死)
        if self.high_confidence_streak >= self.collapse_window or self.echo_chamber_streak >= self.echo_chamber_patience:
            return HealthStatus.COLLAPSED

        # 4. 对局回合死锁与节奏异常 (拖局 > 85 轮或猝死)
        if self.anomalous_streak > 0:
            return HealthStatus.ANOMALOUS

        # 5. 胜率循环克制与大幅震荡
        if self.cycling_streak > 0:
            return HealthStatus.CYCLING

        # 6. 两代断层异动警报 (连续 2 轮胜率 > 75%，警惕基准缺陷或先后手失衡)
        if self.fault_streak >= 2:
            return HealthStatus.BASELINE_FAULT

        # 7. 连续未晋升停滞
        if self.consecutive_failures >= 10:
            return HealthStatus.STAGNANT

        return HealthStatus.HEALTHY

    def _check_termination(
        self, record: IterationRecord, entropy: float, status: HealthStatus
    ) -> Optional[TerminationDecision]:
        """根据全场景异常触发及时的自适应早停判定."""
        # 1. 停滞僵局 (Stagnation Deadlock)
        if self.consecutive_failures >= self.stagnation_patience:
            return TerminationDecision(
                should_terminate=True,
                reason_type="停滞僵局 (Stagnation Deadlock)",
                description=f"连续 {self.consecutive_failures} 轮对抗无法超越基准门禁 (阈值 {self.stagnation_patience} 轮)。当前策略搜索深度已触及瓶颈，继续训练已无法带来正向增益。",
                metric_evidence=f"连续未晋升轮数 = {self.consecutive_failures} | 近期胜率 = {record.win_rate*100:.1f}%",
            )

        # 2. 严重模式塌陷与策略过度锁定 (Severe Mode Collapse / Echo Chamber)
        if self.high_confidence_streak >= self.collapse_window and self.consecutive_failures >= 8:
            return TerminationDecision(
                should_terminate=True,
                reason_type="严重模式塌陷 (Severe Mode Collapse)",
                description="连续多轮 Top-1 准确率高于 97.5% 且胜因多样性香农熵接近 0 (陷入单一抢分套路)，同时连续 8 轮晋升停滞，陷入局部极值纳什陷阱。",
                metric_evidence=f"Top-1 = {record.top1_acc*100:.1f}% | 胜因多样性熵 = {entropy:.3f} | 连续锁定 = {self.high_confidence_streak} 轮",
            )

        if self.echo_chamber_streak >= self.echo_chamber_patience and self.consecutive_failures >= 6:
            return TerminationDecision(
                should_terminate=True,
                reason_type="回音室效应与过度拟合 (Echo Chamber)",
                description="连续多轮训练 Loss 跌破 0.60 且 Top-1 超过 93.0% (丧失探索增量)，网络陷入自博弈同质化死循环，难以产生策略改进。",
                metric_evidence=f"Loss = {record.train_loss:.4f} (<0.60) | Top-1 = {record.top1_acc*100:.1f}% (>93%) | 连续同质 = {self.echo_chamber_streak} 轮",
            )

        # 3. 梯度爆炸与直觉崩溃 (Gradient Instability / Intuition Collapse)
        if self.divergence_streak >= self.divergence_patience:
            return TerminationDecision(
                should_terminate=True,
                reason_type="梯度爆炸与直觉崩溃 (Gradient Instability)",
                description=f"连续 {self.divergence_streak} 轮 Loss 高于 2.00 或 Top-1 低于 70.0%，策略网络直觉崩溃、梯度剧烈震荡，无法正常拟合。",
                metric_evidence=f"最新 Loss = {record.train_loss:.4f} | Top-1 = {record.top1_acc*100:.1f}% | 连续发散 = {self.divergence_streak} 轮",
            )

        # 4. 灾难性遗忘与策略崩溃 (Catastrophic Forgetting)
        if self.degraded_streak >= self.degradation_patience and len(self.history) >= 4:
            normal_history = self.history[: -self.degraded_streak]
            prev_losses = [r.train_loss for r in normal_history[-3:]] if normal_history else []
            avg_prev_loss = sum(prev_losses) / len(prev_losses) if prev_losses else record.train_loss
            if record.win_rate <= 0.15 or record.train_loss > avg_prev_loss * 1.3:
                return TerminationDecision(
                    should_terminate=True,
                    reason_type="灾难性遗忘 (Catastrophic Forgetting)",
                    description=f"连续 {self.degraded_streak} 轮候选模型对抗胜率暴跌至 20% 以下，更新步长过大破坏了已有棋力结构，发生模型策略退化崩溃。",
                    metric_evidence=f"连续崩盘轮数 = {self.degraded_streak} | 最新候选胜率 = {record.win_rate*100:.1f}% | Loss = {record.train_loss:.4f} (基准: {avg_prev_loss:.4f})",
                )

        # 5. 严重策略震荡与循环互克 (Severe Strategy Cycling)
        if self.cycling_streak >= self.cycling_patience and self.consecutive_failures >= 8:
            return TerminationDecision(
                should_terminate=True,
                reason_type="策略循环震荡 (Severe Strategy Cycling)",
                description=f"持续 {self.cycling_streak} 轮陷入大幅胜率震荡与互克循环，模型在局部战术对立面来回摇摆，无法稳定超越基准。",
                metric_evidence=f"连续震荡轮数 = {self.cycling_streak} | 连续未晋升 = {self.consecutive_failures} 轮",
            )

        # 6. 对局死锁与严重节奏异常 (Severe Pace Deadlock)
        if self.anomalous_streak >= self.anomalous_patience:
            return TerminationDecision(
                should_terminate=True,
                reason_type="对局节奏死锁 (Severe Pace Deadlock)",
                description=f"连续 {self.anomalous_streak} 轮对局出现严重长回合对耗或异常短命猝死 (平均回合 > 85 或 < 35)，博弈陷入低效死局。",
                metric_evidence=f"连续异常轮数 = {self.anomalous_streak} | 最新平均回合 = {record.avg_rounds:.1f} 轮",
            )

        return None

    def generate_advice(
        self,
        record: IterationRecord,
        status: HealthStatus,
        decision: Optional[TerminationDecision] = None,
    ) -> AdviceReport:
        """根据异常根因自适应计算下一步超参数调整建议与 CLI 推荐命令."""
        current = record.current_args or {}
        curr_sims = int(current.get("mcts_sims", 30))
        curr_temp = int(current.get("temp_steps", 12))
        curr_eps = float(current.get("dirichlet_eps", 0.25))
        curr_lr = float(current.get("lr", 1e-3))
        curr_buf = int(current.get("buffer_size", 100000))
        curr_epochs = int(current.get("train_epochs", 3))

        root_causes: List[str] = []
        rec_params: Dict[str, Any] = {}

        if status == HealthStatus.COLLAPSED or (decision and ("模式塌陷" in decision.reason_type or "回音室" in decision.reason_type)):
            summary = "模型陷入单一套路纳什陷阱/回音室 (过度确定性与单一胜因)，急需注入高质量探索并扩大历史经验多样性。"
            root_causes.append("策略网络自信度过高 (Top-1 > 93%)，MCTS 先验概率被自身垄断，丧失策略改进增量。")
            root_causes.append("经验回放池内同质局面过多，学习率与训练轮次偏大导致候选模型在单一样本分布上过度拟合。")
            rec_params = {
                "mcts_sims": max(50, curr_sims + 20),
                "temp_steps": max(15, curr_temp + 4),
                "dirichlet_eps": min(0.40, round(curr_eps + 0.10, 2)),
                "lr": max(2e-4, round(curr_lr * 0.4, 5)),
                "buffer_size": max(150000, curr_buf + 50000),
                "train_epochs": max(1, curr_epochs - 1),
            }

        elif status == HealthStatus.EXPLODED or (decision and "梯度爆炸" in decision.reason_type):
            summary = "策略网络直觉崩溃或梯度爆炸，更新步长过激破坏了参数空间，需大幅调低学习率并提高 MCTS 标签质量。"
            root_causes.append("学习率过高 (当前 lr 导致 Loss 飙升 > 2.0) 或网络直觉不稳定 (Top-1 < 70%)。")
            root_causes.append("自博弈推演深度不足以提供可靠监督信号，导致策略头梯度剧烈撕裂。")
            rec_params = {
                "lr": max(1e-4, round(curr_lr * 0.3, 5)),
                "mcts_sims": max(50, curr_sims + 20),
                "train_epochs": max(1, curr_epochs - 1),
                "buffer_size": max(150000, curr_buf + 50000),
            }

        elif status == HealthStatus.STAGNANT or (decision and "停滞僵局" in decision.reason_type):
            summary = "基准模型形成了牢固的防守模式，当前推演深度无法找到更高级破局手，需加深前瞻树推演或切换为深度评估。"
            root_causes.append(f"MCTS 推演次数 ({curr_sims} 次) 难以看清复杂残局与多步兑换价值，导致候选模型无法超越基准。")
            root_causes.append("门禁对抗若采用 PolicyNet 快速评估，容易被表面启发值误导，建议启用深度推演对抗。")
            rec_params = {
                "mcts_sims": max(60, curr_sims * 2),
                "temp_steps": max(14, curr_temp + 2),
                "dirichlet_eps": min(0.35, round(curr_eps + 0.05, 2)),
                "eval_agent": "neural_mcts",
                "lr": round(curr_lr * 0.7, 5),
            }

        elif status == HealthStatus.DEGRADED or (decision and "灾难性遗忘" in decision.reason_type):
            summary = "梯度更新步长过大导致已有棋力结构崩坏，需大幅平滑更新并限制单轮拟合轮次。"
            root_causes.append("学习率与单轮更新 Epochs 偏高，单批自博弈样本的局部噪声剧烈冲刷了网络历史权重。")
            rec_params = {
                "lr": max(1e-4, round(curr_lr * 0.3, 5)),
                "train_epochs": 1,
                "buffer_size": max(150000, curr_buf + 50000),
                "mcts_sims": curr_sims,
            }

        elif status == HealthStatus.CYCLING or (decision and "震荡" in decision.reason_type):
            summary = "候选模型存在策略循环克制 (石头剪刀布效应)，胜率剧烈震荡，需提高经验池混合深度。"
            root_causes.append("ReplayBuffer 滚动窗口过快剔除了历史重要局势，模型在局部战术对立面来回摇摆。")
            rec_params = {
                "buffer_size": max(150000, curr_buf + 50000),
                "lr": round(curr_lr * 0.5, 5),
                "mcts_sims": max(50, curr_sims + 15),
            }

        elif status == HealthStatus.BASELINE_FAULT:
            summary = "两代候选模型发生突然断层 (胜率 > 75%)，通常表明老基准模型存在死锁或评估环境出现黑白方偏倚。"
            root_causes.append("上一代基准模型可能存在特定开局下的走子死锁盲区，被候选模型单方面针对收割。")
            root_causes.append("评估对抗对局样本量偏少或先后手分配不均，建议增加评测对局数并核查先手胜率。")
            rec_params = {
                "games_per_iter": max(50, int(current.get("games_per_iter", 50)) + 20),
                "mcts_sims": max(40, curr_sims),
                "eval_agent": "neural_mcts",
            }

        elif status == HealthStatus.ANOMALOUS or (decision and "死锁" in decision.reason_type):
            summary = "对局出现严重长回合对耗或死锁拖局，需增强先验探索并增加开局随机度以打破防守死局。"
            root_causes.append("双方策略极度保守导致对局逼近百步上限，难以达成有效胜因目标，搜索陷入低效无效推演。")
            rec_params = {
                "temp_steps": max(15, curr_temp + 4),
                "dirichlet_eps": min(0.35, round(curr_eps + 0.08, 2)),
                "mcts_sims": max(50, curr_sims + 15),
                "lr": curr_lr,
            }

        else:
            summary = "自博弈处于健康进化或平稳收官状态，可保持当前参数或提升推演深度冲击更高竞技上限。"
            root_causes.append("模型核心指标 (Loss / Top-1 / 胜率) 运行在绝对健康区间，胜因多样性稳定。")
            rec_params = {
                "mcts_sims": max(50, curr_sims + 10),
                "temp_steps": curr_temp,
                "dirichlet_eps": curr_eps,
                "lr": curr_lr,
            }

        cmd_parts = ["python python/train.py --mode selfplay"]
        cmd_parts.append(f"--iterations {current.get('iterations', 50)}")
        cmd_parts.append(f"--games-per-iter {current.get('games_per_iter', 50)}")
        for k, v in rec_params.items():
            arg_name = k.replace("_", "-")
            cmd_parts.append(f"--{arg_name} {v}")

        rec_cmd = " ".join(cmd_parts)

        return AdviceReport(
            health_status=status,
            summary=summary,
            root_causes=root_causes,
            recommended_params=rec_params,
            recommended_command=rec_cmd,
        )

    def format_step_summary(
        self,
        record: IterationRecord,
        status: HealthStatus,
        decision: Optional[TerminationDecision],
    ) -> str:
        """生成单轮迭代结尾的清晰、三维三色健康指标摘要."""
        entropy = self.compute_shannon_diversity(record.reasons)
        max_h = math.log(3)
        entropy_ratio = (entropy / max_h) * 100 if max_h > 0 else 0.0

        loss_eval = self.evaluate_loss(record.train_loss)
        top1_eval = self.evaluate_top1(record.top1_acc)
        win_eval = self.evaluate_winrate(record.win_rate)

        status_emoji = (
            "🟢"
            if status == HealthStatus.HEALTHY
            else "⚠️"
            if status in [HealthStatus.STAGNANT, HealthStatus.CYCLING, HealthStatus.BASELINE_FAULT]
            else "🚨"
        )

        lines = [
            f"   🩺 [健康指标监测] {status_emoji} 状态: {status.value} | 连续未晋升: {self.consecutive_failures} 轮 | "
            f"胜因熵: {entropy:.3f} (均衡度: {entropy_ratio:.1f}%)",
            f"      📊 核心三维健康度: Loss: {record.train_loss:.3f} {loss_eval.emoji}({loss_eval.tag}) | "
            f"Top-1: {record.top1_acc*100:.1f}% {top1_eval.emoji}({top1_eval.tag}) | "
            f"胜率: {record.win_rate*100:.1f}% {win_eval.emoji}({win_eval.tag})",
        ]

        if decision and decision.should_terminate:
            lines.append(f"   🛑 [早停触发] {decision.reason_type}: {decision.description}")
            lines.append(f"      📌 判定证据: {decision.metric_evidence}")

        return "\n".join(lines)

    def format_terminal_report(
        self,
        decision: Optional[TerminationDecision],
        advice: AdviceReport,
    ) -> str:
        """生成终止或终局时的完整诊断与自适应参数建议看板."""
        banner_title = "🛑 自博弈提前终止告警与破局建议" if decision else "🏁 自博弈全流程总结与进阶调优建议"
        sep = "=" * 80

        lines = [
            "",
            sep,
            f"📋 {banner_title}",
            sep,
            f"🏥 诊断状态: {advice.health_status.value}",
            f"📝 总体结论: {advice.summary}",
        ]

        # 核心指标运行统计与健康区间对照
        if self.history:
            avg_loss = sum(r.train_loss for r in self.history) / len(self.history)
            avg_top1 = sum(r.top1_acc for r in self.history) / len(self.history)
            avg_win = sum(r.win_rate for r in self.history) / len(self.history)
            latest = self.history[-1]

            loss_eval = self.evaluate_loss(latest.train_loss)
            top1_eval = self.evaluate_top1(latest.top1_acc)
            win_eval = self.evaluate_winrate(latest.win_rate)

            lines.extend([
                "",
                "📊 核心三维运行指标对照看板:",
                f"   • Loss 状态    : 最新 {latest.train_loss:.3f} {loss_eval.emoji} | 历史均值 {avg_loss:.3f} | 目标健康区间: [0.90 ~ 1.35] ({loss_eval.comment})",
                f"   • Top-1 直觉   : 最新 {latest.top1_acc*100:.1f}% {top1_eval.emoji} | 历史均值 {avg_top1*100:.1f}% | 目标健康区间: [82.0% ~ 89.0%] ({top1_eval.comment})",
                f"   • 候选胜率     : 最新 {latest.win_rate*100:.1f}% {win_eval.emoji} | 历史均值 {avg_win*100:.1f}% | 目标健康区间: [52.0% ~ 58.0%] ({win_eval.comment})",
            ])

        if decision:
            lines.extend([
                "",
                "🚨 触发早停异常详情:",
                f"   • 异常类型: {decision.reason_type}",
                f"   • 详细描述: {decision.description}",
                f"   • 指标证据: {decision.metric_evidence}",
            ])

        lines.extend([
            "",
            "🔍 深度病灶根因剖析:",
        ])
        for idx, cause in enumerate(advice.root_causes, 1):
            lines.append(f"   {idx}. {cause}")

        lines.extend([
            "",
            "💡 下一阶段推荐自适应超参数配置:",
        ])
        for param, val in advice.recommended_params.items():
            lines.append(f"   • --{param.replace('_', '-')}: {val}")

        lines.extend([
            "",
            "🚀 建议执行的破局重训命令 (直接复制运行):",
            f"   {advice.recommended_command}",
            sep,
            "",
        ])

        return "\n".join(lines)
