"""Training diagnostics, early stopping, and adaptive hyperparameter advisor for Splendor Duel self-play."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class HealthStatus(str, Enum):
    HEALTHY = "正常健康 (Healthy)"
    STAGNANT = "停滞预警 (Stagnant)"
    COLLAPSED = "模式塌陷 (Mode Collapse)"
    DEGRADED = "严重退化 (Degraded)"
    CYCLING = "胜率震荡 (Cycling)"
    ANOMALOUS = "对局异常 (Pace Anomaly)"


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
    ) -> None:
        self.stagnation_patience = stagnation_patience
        self.collapse_window = collapse_window
        self.collapse_top1_thresh = collapse_top1_thresh
        self.collapse_entropy_thresh = collapse_entropy_thresh
        self.degradation_patience = degradation_patience
        self.degradation_winrate_thresh = degradation_winrate_thresh
        self.anomalous_patience = 3
        self.cycling_patience = 6

        self.history: List[IterationRecord] = []
        self.consecutive_failures: int = 0
        self.high_confidence_streak: int = 0
        self.degraded_streak: int = 0
        self.anomalous_streak: int = 0
        self.cycling_streak: int = 0

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
        if record.top1_acc >= self.collapse_top1_thresh and entropy <= self.collapse_entropy_thresh:
            self.high_confidence_streak += 1
        else:
            self.high_confidence_streak = 0

        if record.win_rate <= self.degradation_winrate_thresh:
            self.degraded_streak += 1
        else:
            self.degraded_streak = 0

        # 对局节奏异常判断 (单局走子严重拖局/死循环或异常猝死)
        is_anomalous_pace = record.avg_rounds > 85.0 or (record.avg_rounds > 0 and record.avg_rounds < 35.0)
        if is_anomalous_pace:
            self.anomalous_streak += 1
        else:
            self.anomalous_streak = 0

        # 胜率循环震荡判断 (近 6 轮存在极差超 45% 的两极分化)
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
        # 1. 灾难性退化最优先暴露
        if self.degraded_streak >= self.degradation_patience:
            return HealthStatus.DEGRADED

        # 2. 严重模式塌陷次优先暴露
        if self.high_confidence_streak >= self.collapse_window:
            return HealthStatus.COLLAPSED

        # 3. 对局回合死锁与节奏异常 (优先于停滞曝光，便于及时发现规则走子死循环)
        if self.anomalous_streak > 0:
            return HealthStatus.ANOMALOUS

        # 4. 胜率循环克制与大幅震荡
        if self.cycling_streak > 0:
            return HealthStatus.CYCLING

        # 5. 连续未晋升停滞
        if self.consecutive_failures >= 10:
            return HealthStatus.STAGNANT

        return HealthStatus.HEALTHY

    def _check_termination(
        self, record: IterationRecord, entropy: float, status: HealthStatus
    ) -> Optional[TerminationDecision]:
        """根据 5 大异常场景触发及时的自适应早停判定."""
        # 1. 停滞僵局 (Stagnation Deadlock)
        if self.consecutive_failures >= self.stagnation_patience:
            return TerminationDecision(
                should_terminate=True,
                reason_type="停滞僵局 (Stagnation Deadlock)",
                description=f"连续 {self.consecutive_failures} 轮对抗无法超越基准门禁 (阈值 {self.stagnation_patience} 轮)。当前策略搜索深度已触及瓶颈，继续训练已无法带来正向增益。",
                metric_evidence=f"连续未晋升轮数 = {self.consecutive_failures} | 近期胜率 = {record.win_rate*100:.1f}%",
            )

        # 2. 严重模式塌陷与策略过度锁定 (Severe Mode Collapse)
        if self.high_confidence_streak >= self.collapse_window and self.consecutive_failures >= 8:
            return TerminationDecision(
                should_terminate=True,
                reason_type="严重模式塌陷 (Severe Mode Collapse)",
                description="连续多轮 Top-1 准确率高于 97.5% 且胜因多样性香农熵接近 0 (陷入单一抢分套路)，同时连续 8 轮晋升停滞，陷入局部极值纳什陷阱。",
                metric_evidence=f"Top-1 = {record.top1_acc*100:.1f}% | 胜因多样性熵 = {entropy:.3f} | 连续锁定 = {self.high_confidence_streak} 轮",
            )

        # 3. 灾难性遗忘与策略崩溃 (Catastrophic Forgetting)
        if self.degraded_streak >= self.degradation_patience and len(self.history) >= 4:
            # 获取崩盘前的正常历史轮次 Loss 均值作为基准
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

        # 4. 严重策略震荡与循环互克 (Severe Strategy Cycling)
        if self.cycling_streak >= self.cycling_patience and self.consecutive_failures >= 8:
            return TerminationDecision(
                should_terminate=True,
                reason_type="策略循环震荡 (Severe Strategy Cycling)",
                description=f"持续 {self.cycling_streak} 轮陷入大幅胜率震荡与互克循环，模型在局部战术对立面来回摇摆，无法稳定超越基准。",
                metric_evidence=f"连续震荡轮数 = {self.cycling_streak} | 连续未晋升 = {self.consecutive_failures} 轮",
            )

        # 5. 对局死锁与严重节奏异常 (Severe Pace Deadlock)
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

        if status == HealthStatus.COLLAPSED or (decision and "模式塌陷" in decision.reason_type):
            summary = "模型陷入单一套路纳什陷阱 (过度确定性与单一胜因)，急需重新注入高质量探索与扩大历史经验多样性。"
            root_causes.append("策略网络自信度过高 (Top-1 > 97%)，MCTS 先验概率被自身垄断，无法发掘皇冠或封锁单色等替代路径。")
            root_causes.append("经验回放池内同质局面过多，学习率过高导致候选模型在单一样本分布上快速塌陷。")
            rec_params = {
                "mcts_sims": max(50, curr_sims + 20),
                "temp_steps": max(15, curr_temp + 4),
                "dirichlet_eps": min(0.40, round(curr_eps + 0.10, 2)),
                "lr": max(2e-4, round(curr_lr * 0.4, 5)),
                "buffer_size": max(150000, curr_buf + 50000),
                "train_epochs": max(1, curr_epochs - 1),
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
            root_causes.append("模型胜因多样性与门禁对抗胜率保持稳定，无明显死锁或塌陷。")
            rec_params = {
                "mcts_sims": max(50, curr_sims + 10),
                "temp_steps": curr_temp,
                "dirichlet_eps": curr_eps,
                "lr": curr_lr,
            }

        # 构建可直接复制执行的推荐 CLI 命令
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
        """生成单轮迭代结尾的紧凑健康指标摘要."""
        entropy = self.compute_shannon_diversity(record.reasons)
        max_h = math.log(3)
        entropy_ratio = (entropy / max_h) * 100 if max_h > 0 else 0.0

        status_emoji = "🟢" if status == HealthStatus.HEALTHY else "⚠️" if status in [HealthStatus.STAGNANT, HealthStatus.CYCLING] else "🚨"

        lines = [
            f"   🩺 [健康指标监测] {status_emoji} 状态: {status.value} | 胜因多样性熵: {entropy:.3f} (均衡度: {entropy_ratio:.1f}%) | "
            f"连续未晋升: {self.consecutive_failures} 轮 | Top-1: {record.top1_acc*100:.1f}%"
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
