"""Unit tests for TrainingAdvisor, anomaly detection, early stopping, and adaptive advisor."""

import math
import pytest
from splendor_ai.advisor import (
    AdviceReport,
    HealthStatus,
    IterationRecord,
    TerminationDecision,
    TrainingAdvisor,
)


def _make_dummy_record(
    iteration: int = 1,
    train_loss: float = 0.5,
    top1_acc: float = 0.85,
    win_rate: float = 0.50,
    promoted: bool = False,
    reasons: dict = None,
    avg_rounds: float = 60.0,
    lr: float = 1e-3,
) -> IterationRecord:
    if reasons is None:
        reasons = {"20_points": 10, "10_crowns": 5, "10_color_points": 5}
    return IterationRecord(
        iteration=iteration,
        train_loss=train_loss,
        policy_loss=train_loss * 0.7,
        value_loss=train_loss * 0.3,
        top1_acc=top1_acc,
        top3_acc=0.95,
        win_rate=win_rate,
        promoted=promoted,
        candidate_wins=int(win_rate * 20),
        baseline_wins=20 - int(win_rate * 20),
        draws=0,
        reasons=reasons,
        avg_rounds=avg_rounds,
        avg_steps=avg_rounds * 2.6,
        lr=lr,
        samples_added=8000,
        buffer_size=50000,
        current_args={"mcts_sims": 30, "temp_steps": 12, "dirichlet_eps": 0.25, "lr": lr},
    )


def test_shannon_diversity_index():
    # 1. 完全单一胜因 -> 熵为 0
    single_reason = {"20_points": 20, "10_crowns": 0, "10_color_points": 0}
    h_single = TrainingAdvisor.compute_shannon_diversity(single_reason)
    assert math.isclose(h_single, 0.0, abs_tol=1e-6)

    # 2. 三大胜因完全均分 -> 最大熵 ln(3) ≈ 1.0986
    equal_reasons = {"20_points": 10, "10_crowns": 10, "10_color_points": 10}
    h_equal = TrainingAdvisor.compute_shannon_diversity(equal_reasons)
    assert math.isclose(h_equal, math.log(3), abs_tol=1e-4)

    # 3. 空胜因字典
    assert TrainingAdvisor.compute_shannon_diversity({}) == 0.0


def test_healthy_progression():
    advisor = TrainingAdvisor()
    for i in range(1, 6):
        rec = _make_dummy_record(
            iteration=i,
            train_loss=0.6 - i * 0.05,
            top1_acc=0.82 + i * 0.02,
            win_rate=0.60 if i % 2 == 0 else 0.45,
            promoted=(i % 2 == 0),
        )
        status, decision, advice = advisor.step(rec)
        assert status == HealthStatus.HEALTHY
        assert decision is None
        assert advice is None


def test_stagnation_early_stop():
    # 连续 15 轮未晋升，触发 Stagnation 及时终止
    advisor = TrainingAdvisor(stagnation_patience=15)
    for i in range(1, 15):
        rec = _make_dummy_record(iteration=i, promoted=False, win_rate=0.45)
        status, decision, advice = advisor.step(rec)
        assert decision is None

    # 第 15 轮未晋升 -> 触发早停
    rec15 = _make_dummy_record(iteration=15, promoted=False, win_rate=0.40)
    status, decision, advice = advisor.step(rec15)
    assert decision is not None
    assert decision.should_terminate is True
    assert "停滞僵局" in decision.reason_type
    assert advice is not None
    assert advice.recommended_params["mcts_sims"] >= 60
    assert "--eval-agent neural_mcts" in advice.recommended_command


def test_mode_collapse_early_stop():
    # 模拟模式塌陷：Top-1 达到 98.5%，胜因 100% 为声望，且连续 8 轮未晋升
    advisor = TrainingAdvisor(collapse_window=6)
    single_reason = {"20_points": 20, "10_crowns": 0, "10_color_points": 0}

    for i in range(1, 9):
        rec = _make_dummy_record(
            iteration=i,
            top1_acc=0.985,
            promoted=False,
            win_rate=0.45,
            reasons=single_reason,
        )
        status, decision, advice = advisor.step(rec)

    assert status == HealthStatus.COLLAPSED
    assert decision is not None
    assert decision.should_terminate is True
    assert "严重模式塌陷" in decision.reason_type
    assert advice is not None
    # 建议应该增加 temp_steps 和 dirichlet_eps，并降低 lr
    assert advice.recommended_params["temp_steps"] > 12
    assert advice.recommended_params["dirichlet_eps"] > 0.25
    assert advice.recommended_params["lr"] < 1e-3


def test_catastrophic_forgetting_early_stop():
    # 模拟连续 3 轮候选胜率 <= 15%，且 Loss 攀升
    advisor = TrainingAdvisor(degradation_patience=3)

    # 前序正常 3 轮
    for i in range(1, 4):
        advisor.step(_make_dummy_record(iteration=i, train_loss=0.30, win_rate=0.55, promoted=True))

    # 接着连续 3 轮崩盘
    advisor.step(_make_dummy_record(iteration=4, train_loss=0.45, win_rate=0.15, promoted=False))
    advisor.step(_make_dummy_record(iteration=5, train_loss=0.50, win_rate=0.10, promoted=False))
    status, decision, advice = advisor.step(
        _make_dummy_record(iteration=6, train_loss=0.55, win_rate=0.10, promoted=False)
    )

    assert status == HealthStatus.DEGRADED
    assert decision is not None
    assert decision.should_terminate is True
    assert "灾难性遗忘" in decision.reason_type
    assert advice is not None
    assert advice.recommended_params["train_epochs"] == 1
    assert advice.recommended_params["lr"] <= 3e-4


def test_cycling_and_pace_anomalies():
    # 1. 胜率剧烈震荡 (Cycling) 识别与早停
    advisor_cycle = TrainingAdvisor()
    # 先进行 5 轮未晋升铺垫
    for i in range(1, 6):
        advisor_cycle.step(_make_dummy_record(iteration=i, win_rate=0.45, promoted=False))
    # 接着模拟两极大幅震荡 (且均未超过 0.55 晋升门禁，持续至达到早停阈值)
    rates = [0.60, 0.15, 0.60, 0.15, 0.60, 0.15, 0.60]
    decision = None
    for i, r in enumerate(rates, 6):
        status, decision, advice = advisor_cycle.step(
            _make_dummy_record(iteration=i, win_rate=r, promoted=False)
        )
    assert status == HealthStatus.CYCLING
    assert decision is not None
    assert decision.should_terminate is True
    assert "策略循环震荡" in decision.reason_type
    assert advice is not None

    # 2. 异常长回合死锁拖局 (Pace Deadlock) 识别与早停
    advisor_pace = TrainingAdvisor()
    # 连续 2 轮异常
    for i in range(1, 3):
        status, decision, _ = advisor_pace.step(_make_dummy_record(iteration=i, avg_rounds=95.0, promoted=False))
        assert status == HealthStatus.ANOMALOUS
        assert decision is None
    # 第 3 轮触发死锁早停
    status, decision, advice = advisor_pace.step(_make_dummy_record(iteration=3, avg_rounds=92.0, promoted=False))
    assert status == HealthStatus.ANOMALOUS
    assert decision is not None
    assert decision.should_terminate is True
    assert "对局节奏死锁" in decision.reason_type
    assert advice is not None
    assert advice.recommended_params["temp_steps"] > 12


def test_report_formatting():
    advisor = TrainingAdvisor()
    rec = _make_dummy_record(iteration=1, promoted=True)
    status, decision, advice = advisor.step(rec)

    # 格式化单行摘要
    summary_str = advisor.format_step_summary(rec, status, decision)
    assert "健康指标监测" in summary_str
    assert "正常健康" in summary_str

    # 格式化终局看板
    adv = advisor.generate_advice(rec, status)
    terminal_str = advisor.format_terminal_report(None, adv)
    assert "自博弈全流程总结与进阶调优建议" in terminal_str
    assert "建议执行的破局重训命令" in terminal_str
    assert "python python/train.py --mode selfplay" in terminal_str
