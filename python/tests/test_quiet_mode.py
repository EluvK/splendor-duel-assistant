"""Unit tests for quiet-mode resource limits and CLI arguments."""

import os
import sys
from unittest.mock import patch
import torch

from train import configure_resource_limits, parse_args, set_low_process_priority


def test_cli_quite_mode_and_quiet_mode():
    """验证 --quite-mode 与 --quiet-mode 均能正确激活静默模式，且不改动任何训练效果超参数."""
    # 1. 测试 --quite-mode (用户习惯写法)
    with patch.object(sys, "argv", ["train.py", "--mode", "selfplay", "--quite-mode"]):
        args = parse_args()
        assert args.quiet_mode is True
        # 验证核心质量超参保持完整默认值
        assert args.mcts_sims == 100
        assert args.games_per_iter == 60
        assert args.iterations == 30
        assert args.train_epochs == 4
        assert args.promote_threshold == 0.53

    # 2. 测试 --quiet-mode (标准写法)
    with patch.object(sys, "argv", ["train.py", "--mode", "selfplay", "--quiet-mode"]):
        args = parse_args()
        assert args.quiet_mode is True
        assert args.mcts_sims == 100

    # 3. 测试默认情况
    with patch.object(sys, "argv", ["train.py", "--mode", "selfplay"]):
        args = parse_args()
        assert args.quiet_mode is False


def test_cli_cpu_threads_argument():
    """验证可显式自定义 cpu-threads 参数."""
    with patch.object(sys, "argv", ["train.py", "--mode", "selfplay", "--cpu-threads", "4"]):
        args = parse_args()
        assert args.cpu_threads == 4


def test_configure_resource_limits_quiet():
    """验证 quiet_mode 下分配的线程数为总核心数的一半，且环境变量与 PyTorch 线程设置同步."""
    total_cpus = os.cpu_count() or 4
    expected_half = max(1, total_cpus // 2)

    res = configure_resource_limits(quiet_mode=True)
    assert res["quiet_mode"] is True
    assert res["allocated_threads"] == expected_half
    assert os.environ["RAYON_NUM_THREADS"] == str(expected_half)
    assert os.environ["OMP_NUM_THREADS"] == str(expected_half)
    assert torch.get_num_threads() == expected_half


def test_configure_resource_limits_full_speed():
    """验证全速模式下分配全部计算核心."""
    total_cpus = os.cpu_count() or 4
    res = configure_resource_limits(quiet_mode=False)
    assert res["quiet_mode"] is False
    assert res["allocated_threads"] == total_cpus
    assert os.environ["RAYON_NUM_THREADS"] == str(total_cpus)
    assert torch.get_num_threads() == total_cpus


def test_set_low_process_priority():
    """验证降低进程优先级函数正常执行不抛异常."""
    result = set_low_process_priority()
    assert isinstance(result, bool)
