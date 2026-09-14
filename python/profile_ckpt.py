"""Checkpoint profiler and diagnostic benchmark for Splendor Duel."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import torch

from splendor_ai._engine import evaluate_neural_match
from splendor_ai.arena import Arena
from splendor_ai.mcts import HeuristicAgent, MCTSAgent, NeuralMCTSAgent, PolicyNetAgent
from splendor_ai.net import SplendorNet


def inspect_metadata(ckpt_path: Path) -> Dict[str, Any]:
    """读取并解析 checkpoint 元数据与架构信息."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt_path}")

    stat = ckpt_path.stat()
    file_size_mb = stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt.get("epoch", 0)
    meta = ckpt.get("meta", {})
    iteration = meta.get("iteration", None)
    win_rate = meta.get("win_rate", None)
    promoted = meta.get("promoted", None)
    c_wins = meta.get("candidate_wins", None)
    b_wins = meta.get("baseline_wins", None)
    total_games = meta.get("total_games", None)
    total_samples = meta.get("total_samples", None)

    # 模型参数量统计
    net = SplendorNet()
    net.load_state_dict(ckpt["model_state"])
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return {
        "path": str(ckpt_path),
        "size_mb": file_size_mb,
        "mod_time": mod_time,
        "epoch": epoch,
        "iteration": iteration,
        "win_rate": win_rate,
        "promoted": promoted,
        "candidate_wins": c_wins,
        "baseline_wins": b_wins,
        "total_games": total_games,
        "total_samples": total_samples,
        "meta": meta,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "net": net,
    }


def run_benchmark(
    ckpt_path: str = "checkpoints/best.pt",
    games: int = 10,
    device_str: str = "auto",
    use_mcts: bool = False,
    mcts_sims: int = 30,
    workers: int = 0,
    backend: str = "rust",
) -> None:
    """运行 Checkpoint 全景评估诊断套件."""
    path = Path(ckpt_path)
    info = inspect_metadata(path)
    net: SplendorNet = info["net"]

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    net.to(device)
    net.eval()

    num_pairs = max(1, games // 2)
    actual_games = num_pairs * 2

    # 1. 打印档案元数据
    print("=" * 80)
    print("🔍 《璀璨宝石：对决》Checkpoint 档案与诊断评估")
    print("=" * 80)
    print(f"📁 权重路径: {info['path']}")
    print(f"📦 文件大小: {info['size_mb']:.2f} MB | 修改时间: {info['mod_time']}")
    print(f"🧠 模型参数: {info['total_params']:,} 个 (可训: {info['trainable_params']:,})")
    print(f"⚙️  运行设备: {device.type.upper()} | 评估引擎: {backend.upper()}")

    print("\n📜 训练血统与经历数据:")
    if info["iteration"] is not None:
        print(f"   • 自博弈迭代轮次: Iteration {info['iteration']}")
    print(f"   • 累计训练 Epochs: {info['epoch']}")
    if info["win_rate"] is not None:
        promo_str = f"   • 晋升对抗战绩: 胜率 {info['win_rate']*100:.1f}%"
        if info["candidate_wins"] is not None:
            promo_str += f" ({info['candidate_wins']} 胜 / {info['baseline_wins']} 负)"
        print(promo_str)

    # 统计数据量
    if info["total_games"] is not None:
        sample_str = f" ({info['total_samples']:,} 真实决策步)" if info["total_samples"] is not None else ""
        print(f"   • 累计训练经历: {info['total_games']:,} 局对弈{sample_str}")
    elif info["iteration"]:
        est_games = info["iteration"] * 100
        print(f"   • 累计自博弈经历 (估算): 约 {est_games:,} 局对弈 (~{est_games * 160:,} 决策步)")
    elif "train" in info["meta"]:
        print(f"   • 模仿学习最后指标: loss={info['meta']['train'].get('loss', 0):.4f}")

    agent_desc = f"NeuralMCTS-{mcts_sims}" if use_mcts else "PolicyNet"

    # Rust 高速并行后端模式 (秒级极速 MCTS 与直觉网络对弈)
    if backend == "rust":
        onnx_bytes = net.export_onnx_bytes()
        sims = mcts_sims if use_mcts else 0

        print("-" * 80)
        print(f"⚔️ [基准 1] 自我对战评估 (Self vs Self) - {actual_games} 局成对对抗 (决策: {agent_desc} | 后端: Rust 8核并发)")
        t0 = time.time()
        total_g, a0_wins, a1_wins, draws, reasons_self = evaluate_neural_match(
            onnx_bytes,
            onnx_bytes,
            num_pairs=num_pairs,
            base_seed=42,
            num_sims=sims,
        )
        dur_self = time.time() - t0
        p0_wins = reasons_self.get("p0_seat_wins", 0)
        p1_wins = reasons_self.get("p1_seat_wins", 0)
        avg_steps_self = reasons_self.get("total_steps", 0) / max(actual_games, 1)
        avg_rounds_self = reasons_self.get("total_rounds", 0) / max(actual_games, 1)
        print(f"   ⏱️  对战耗时: {dur_self:.2f}s (平均每局 {dur_self/actual_games:.2f}s)")
        print(f"   📊 平均对局长度: {avg_rounds_self:.1f} 轮 (共 {avg_steps_self:.1f} 动作步)")
        print(
            f"   ⚖️  先后手平衡性: 先手(P0) 胜率 {p0_wins/max(actual_games, 1)*100:.1f}% | 后手(P1) 胜率 {p1_wins/max(actual_games, 1)*100:.1f}%"
        )
        if reasons_self:
            print(
                f"   🎯 终局胜因统计: 20声望胜 {reasons_self.get('20_points', 0)} 局 | "
                f"10皇冠胜 {reasons_self.get('10_crowns', 0)} 局 | "
                f"10单色胜 {reasons_self.get('10_color_points', 0)} 局"
            )

        # 2. 对战启发式 AI
        print("\n" + "-" * 80)
        print(f"🥊 [基准 2] 对战启发式 AI (Model vs HeuristicAI) - {actual_games} 局成对对抗 (消除发牌偏差 | 后端: Rust 8核并发)")
        t1 = time.time()
        total_heu, heu_m_wins, heu_ai_wins, heu_draws, reasons_heu = evaluate_neural_match(
            onnx_bytes,
            None,
            num_pairs=num_pairs,
            base_seed=2024,
            num_sims=sims,
        )
        dur_heu = time.time() - t1
        avg_steps_heu = reasons_heu.get("total_steps", 0) / max(actual_games, 1)
        avg_rounds_heu = reasons_heu.get("total_rounds", 0) / max(actual_games, 1)
        win_steps = reasons_heu.get("agent0_win_steps", 0) / max(heu_m_wins, 1)
        win_rounds = reasons_heu.get("agent0_win_rounds", 0) / max(heu_m_wins, 1)
        lose_steps = reasons_heu.get("agent0_lose_steps", 0) / max(heu_ai_wins, 1)
        lose_rounds = reasons_heu.get("agent0_lose_rounds", 0) / max(heu_ai_wins, 1)

        win_rounds_str = (
            f"{win_rounds:.1f} 轮 ({win_steps:.1f} 步)"
            if heu_m_wins > 0
            else "无胜场"
        )
        lose_rounds_str = (
            f"{lose_rounds:.1f} 轮 ({lose_steps:.1f} 步)"
            if heu_ai_wins > 0
            else "全胜未尝一败"
        )
        print(f"   ⏱️  对战耗时: {dur_heu:.2f}s (平均每局 {dur_heu/actual_games:.2f}s)")
        print(
            f"   🏆 胜负总览: 模型胜 {heu_m_wins} 局 | 启发式胜 {heu_ai_wins} 局 | 平局 {heu_draws} 局"
        )
        print(f"   📈 对抗胜率: {heu_m_wins/max(actual_games, 1)*100:.1f}%")
        print(f"   ✨ 赢下来平均花费: {win_rounds_str}")
        print(f"   🛡️  输掉时平均坚持: {lose_rounds_str}")
        print(f"   📊 全场平均长度: {avg_rounds_heu:.1f} 轮 (共 {avg_steps_heu:.1f} 动作步)")
        if reasons_heu:
            print(
                f"   🎯 终局胜因统计: 20声望胜 {reasons_heu.get('20_points', 0)} 局 | "
                f"10皇冠胜 {reasons_heu.get('10_crowns', 0)} 局 | "
                f"10单色胜 {reasons_heu.get('10_color_points', 0)} 局"
            )
        print("=" * 80 + "\n")
        return

    # Python Arena 传统评估模式
    agent_eval1 = (
        NeuralMCTSAgent(net, device, num_sims=mcts_sims) if use_mcts else PolicyNetAgent(net, device)
    )
    agent_eval2 = (
        NeuralMCTSAgent(net, device, num_sims=mcts_sims) if use_mcts else PolicyNetAgent(net, device)
    )

    print("-" * 80)
    print(f"⚔️ [基准 1] 自我对战评估 (Self vs Self) - {actual_games} 局成对对抗 (决策: {agent_desc})")
    arena_self = Arena(
        agent_eval1,
        agent_eval2,
        agent0_name=f"{agent_desc}-Me",
        agent1_name=f"{agent_desc}-Mirror",
    )
    t0 = time.time()
    res_self = arena_self.play_match(num_pairs=num_pairs, base_seed=42, workers=workers)
    dur_self = time.time() - t0

    print(f"   ⏱️  对战耗时: {dur_self:.2f}s (平均每局 {dur_self/actual_games:.2f}s)")
    print(f"   📊 平均对局长度: {res_self.avg_rounds:.1f} 轮 (共 {res_self.avg_steps:.1f} 动作步)")
    print(
        f"   ⚖️  先后手平衡性: 先手(P0) 胜率 {res_self.p0_seat_win_rate*100:.1f}% | 后手(P1) 胜率 {res_self.p1_seat_win_rate*100:.1f}%"
    )
    if res_self.reasons:
        print(
            f"   🎯 终局胜因统计: 20声望胜 {res_self.reasons.get('20_points', 0)} 局 | "
            f"10皇冠胜 {res_self.reasons.get('10_crowns', 0)} 局 | "
            f"10单色胜 {res_self.reasons.get('10_color_points', 0)} 局"
        )

    # 2. 对战启发式 AI
    print("\n" + "-" * 80)
    print(f"🥊 [基准 2] 对战启发式 AI (Model vs HeuristicAI) - {actual_games} 局成对对抗 (消除发牌偏差)")
    heuristic_agent = HeuristicAgent()
    arena_heu = Arena(
        agent_eval1,
        heuristic_agent,
        agent0_name=agent_desc,
        agent1_name="HeuristicAI",
    )
    t1 = time.time()
    res_heu = arena_heu.play_match(num_pairs=num_pairs, base_seed=2024, workers=workers)
    dur_heu = time.time() - t1

    print(f"   ⏱️  对战耗时: {dur_heu:.2f}s (平均每局 {dur_heu/actual_games:.2f}s)")
    print(
        f"   🏆 胜负总览: 模型胜 {res_heu.agent0_wins} 局 | 启发式胜 {res_heu.agent1_wins} 局 | 平局 {res_heu.draws} 局"
    )
    print(f"   📈 对抗胜率: {res_heu.agent0_win_rate*100:.1f}%")

    # 核心轮数与步数统计：赢下来花多少轮/步 vs 输掉时坚持多少轮/步
    win_rounds_str = (
        f"{res_heu.avg_win_rounds:.1f} 轮 ({res_heu.avg_win_steps:.1f} 步)"
        if res_heu.agent0_wins > 0
        else "无胜场"
    )
    lose_rounds_str = (
        f"{res_heu.avg_lose_rounds:.1f} 轮 ({res_heu.avg_lose_steps:.1f} 步)"
        if res_heu.agent1_wins > 0
        else "全胜未尝一败"
    )
    print(f"   ✨ 赢下来平均花费: {win_rounds_str}")
    print(f"   🛡️  输掉时平均坚持: {lose_rounds_str}")
    print(f"   📊 全场平均长度: {res_heu.avg_rounds:.1f} 轮 (共 {res_heu.avg_steps:.1f} 动作步)")
    if res_heu.reasons:
        print(
            f"   🎯 终局胜因统计: 20声望胜 {res_heu.reasons.get('20_points', 0)} 局 | "
            f"10皇冠胜 {res_heu.reasons.get('10_crowns', 0)} 局 | "
            f"10单色胜 {res_heu.reasons.get('10_color_points', 0)} 局"
        )

    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Splendor Duel Checkpoint Profiler")
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default="checkpoints/best.pt",
        help="Path to the checkpoint file (default: checkpoints/best.pt)",
    )
    parser.add_argument("--games", type=int, default=20, help="Total evaluation games for each benchmark (default: 20)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--mcts", action="store_true", help="Evaluate with MCTS search instead of pure PolicyNet")
    parser.add_argument("--sims", type=int, default=30, help="MCTS simulation count if --mcts is enabled")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker threads for arena evaluation (default: 0 for auto)")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rust", "python"],
        default="rust",
        help="Evaluation backend: 'rust' (recommended, 8-thread C++ ONNX engine) or 'python'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        ckpt_path=args.checkpoint,
        games=args.games,
        device_str=args.device,
        use_mcts=args.mcts,
        mcts_sims=args.sims,
        workers=args.workers,
        backend=args.backend,
    )
