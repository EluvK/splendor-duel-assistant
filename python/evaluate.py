"""Independent arena evaluation CLI tool for Splendor Duel.

Supports:
- Checkpoint vs Checkpoint (e.g. iter_216 vs iter_060) with MCTS or PolicyNet
- Checkpoint vs HeuristicAI with MCTS or PolicyNet
- Ultra-fast multi-threaded Rust evaluation (8-core Rayon ONNX)
- Legacy Python Arena mode with step-by-step trace
"""

import argparse
from pathlib import Path
import time
from typing import Optional, Tuple
import torch

from splendor_ai._engine import evaluate_neural_match
from splendor_ai.arena import Arena
from splendor_ai.mcts import (
    Agent,
    HeuristicAgent,
    MCTSAgent,
    NeuralMCTSAgent,
    PolicyNetAgent,
    RandomAgent,
    RustMCTSAgent,
)
from splendor_ai.net import SplendorNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Splendor Duel Arena Match Evaluator")
    parser.add_argument(
        "--profile",
        type=str,
        nargs="?",
        const="checkpoints/best.pt",
        default=None,
        help="Quickly profile and diagnose a checkpoint (e.g. --profile checkpoints/best.pt)",
    )
    # 核心快速评测参数：支持两个权重直接对决
    parser.add_argument(
        "--model1",
        type=str,
        default="checkpoints/best.pt",
        help="Path to checkpoint 1 (e.g. checkpoints/iter_216.pt or checkpoints/best.pt)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Path to checkpoint 2 (e.g. checkpoints/iter_060.pt). If omitted or 'heuristic', plays against HeuristicAI",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=30,
        help="MCTS simulation count (0 for pure PolicyNet, >0 for Neural MCTS)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=10,
        help="Paired match count (total games = 2 * pairs, strict seat swap)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rust", "python"],
        default="rust",
        help="Evaluation backend: 'rust' (recommended, 8-thread C++ speed) or 'python'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cuda', 'cpu')",
    )

    # 传统细粒度 Agent 类型（仅当使用 Python 引擎或非神经网络 Agent 时）
    parser.add_argument(
        "--agent1",
        type=str,
        choices=["neural_mcts", "net", "rust_mcts", "mcts", "heuristic", "random"],
        default=None,
    )
    parser.add_argument(
        "--agent2",
        type=str,
        choices=["neural_mcts", "net", "rust_mcts", "mcts", "heuristic", "random"],
        default=None,
    )

    return parser.parse_args()


def load_net_bytes(model_path: str, device: torch.device) -> Tuple[SplendorNet, bytes, str]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {model_path}")

    net = SplendorNet().to(device)
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model_state"])
    onnx_bytes = net.export_onnx_bytes()

    meta = ckpt.get("meta", {})
    it = meta.get("iteration")
    desc = f"{path.stem}" + (f" (Iter {it})" if it is not None else "")
    return net, onnx_bytes, desc


def run_rust_eval(
    model1_path: str,
    model2_path: Optional[str],
    sims: int,
    pairs: int,
    device: torch.device,
) -> None:
    """使用底层纯 Rust 多线程 8 核并发引擎运行极速对决 (支持 MCTS 树搜索)."""
    net1, bytes1, name1 = load_net_bytes(model1_path, device)
    agent1_name = f"{name1} [MCTS-{sims}]" if sims > 0 else f"{name1} [PolicyNet]"

    bytes2 = None
    if model2_path is not None and model2_path.lower() not in ["none", "heuristic", "heuristic_ai"]:
        net2, bytes2, name2 = load_net_bytes(model2_path, device)
        agent2_name = f"{name2} [MCTS-{sims}]" if sims > 0 else f"{name2} [PolicyNet]"
    else:
        agent2_name = "HeuristicAI (内置专家规则)"

    total_games = pairs * 2
    mode_desc = f"Neural MCTS (推演: {sims} 次/步)" if sims > 0 else "纯直觉网络 (0 次推演)"
    print("\n" + "=" * 75)
    print(f"⚔️ 《璀璨宝石：对决》全速多线程成对对抗评测")
    print(f"   • 选手 1: {agent1_name}")
    print(f"   • 选手 2: {agent2_name}")
    print(f"   • 决策引擎: {mode_desc} | 换座轮数: {pairs} 对 (共 {total_games} 局)")
    print("=" * 75)

    t0 = time.time()
    base_seed = int(time.time()) & 0x7FFFFFFF
    total_g, a1_wins, a2_wins, draws, reasons = evaluate_neural_match(
        bytes1,
        bytes2,
        num_pairs=pairs,
        base_seed=base_seed,
        num_sims=sims,
    )
    elapsed = time.time() - t0

    wr1 = a1_wins / max(total_g, 1)
    wr2 = a2_wins / max(total_g, 1)

    avg_steps = reasons.get("total_steps", 0) / max(total_g, 1)
    avg_rounds = reasons.get("total_rounds", 0) / max(total_g, 1)

    print(f"\n📊 【比赛结果终报】(耗时 {elapsed:.2f}s | 速度: {total_g/max(elapsed, 1e-6):.1f} 局/秒 | 平均: {avg_rounds:.1f} 轮 / {avg_steps:.1f} 步)")
    print("-" * 75)
    print(f"🥇 {agent1_name:<38} 胜场: {a1_wins:>3} 局 ({wr1*100:5.1f}%)")
    print(f"🥈 {agent2_name:<38} 胜场: {a2_wins:>3} 局 ({wr2*100:5.1f}%)")
    if draws > 0:
        print(f"🤝 平局: {draws} 局")
    print("-" * 75)

    p0_wins = reasons.get("p0_seat_wins", 0)
    p1_wins = reasons.get("p1_seat_wins", 0)
    print(f"⚖️  座位优势分析: 先手(P0) 胜率 {p0_wins/max(total_g, 1)*100:.1f}% | 后手(P1) 胜率 {p1_wins/max(total_g, 1)*100:.1f}%")

    r20 = reasons.get("20_points", 0)
    r_crown = reasons.get("10_crowns", 0)
    r_col = reasons.get("10_color_points", 0)
    print(f"🎯 胜因构成统计: 20声望胜 {r20} 局 ({r20/max(total_g, 1)*100:.0f}%) | "
          f"10皇冠胜 {r_crown} 局 ({r_crown/max(total_g, 1)*100:.0f}%) | "
          f"10单色胜 {r_col} 局 ({r_col/max(total_g, 1)*100:.0f}%)")
    print("=" * 75 + "\n")


def load_python_agent(
    agent_type: Optional[str],
    model_path: Optional[str],
    sims: int,
    device: torch.device,
) -> Tuple[Agent, str]:
    if agent_type == "heuristic" or (agent_type is None and model_path is None):
        return HeuristicAgent(), "HeuristicAI"
    if agent_type == "random":
        return RandomAgent(), "RandomAI"
    if agent_type in ["rust_mcts", "mcts"]:
        return MCTSAgent(num_sims=sims), f"HeuristicMCTS-{sims}"

    # 神经网络
    net = SplendorNet().to(device)
    name = Path(model_path).stem if model_path else "Net"
    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        net.load_state_dict(ckpt["model_state"])

    if agent_type == "net" or (agent_type is None and sims == 0):
        return PolicyNetAgent(net, device), f"Policy({name})"
    else:
        return NeuralMCTSAgent(net, device, num_sims=sims), f"NeuralMCTS-{sims}({name})"


def main() -> None:
    args = parse_args()

    if args.profile:
        from profile_ckpt import run_benchmark
        run_benchmark(
            ckpt_path=args.profile,
            games=args.pairs * 2,
            device_str=args.device,
            use_mcts=args.sims > 0,
            mcts_sims=args.sims,
        )
        return

    device_str = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    device = torch.device(device_str)

    # 如果未显式指定 agent 类型且选择 rust 后端，走全速并行 Rust 引擎
    is_custom_agent = (args.agent1 is not None and args.agent1 not in ["neural_mcts", "net"]) or \
                      (args.agent2 is not None and args.agent2 not in ["neural_mcts", "net", "heuristic"])
    if args.backend == "rust" and not is_custom_agent:
        run_rust_eval(
            model1_path=args.model1,
            model2_path=args.model2,
            sims=args.sims,
            pairs=args.pairs,
            device=device,
        )
        return

    # Python Arena 模式
    agent1, name1 = load_python_agent(args.agent1, args.model1, args.sims, device)
    agent2, name2 = load_python_agent(args.agent2, args.model2, args.sims, device)

    print("=" * 60)
    print(f"⚔️ 《璀璨宝石：对决》Python Arena 模拟对弈")
    print(f"   • {name1} vs {name2} ({args.pairs * 2} 局成对对抗)")
    print("=" * 60)

    arena = Arena(agent1, agent2, agent0_name=name1, agent1_name=name2)
    t0 = time.time()
    res = arena.play_match(num_pairs=args.pairs, base_seed=int(time.time()))
    dur = time.time() - t0

    print("\n================== 比赛结果战报 ==================")
    print(f"耗时: {dur:.1f}s | 总局数: {res.total_games} 局 | 平均长度: {res.avg_rounds:.1f} 轮 (共 {res.avg_steps:.1f} 步)")
    print("-" * 50)
    print(f"🥇 {res.agent0_name:<25} 胜场: {res.agent0_wins:>3} 局 ({res.agent0_win_rate*100:5.1f}%)")
    print(f"🥈 {res.agent1_name:<25} 胜场: {res.agent1_wins:>3} 局 ({(1.0 - res.agent0_win_rate)*100:5.1f}%)")
    if res.draws > 0:
        print(f"🤝 平局: {res.draws} 局")
    print("-" * 50)
    print(f"{res.agent0_name} 作为先手 (P0) 胜场: {res.agent0_as_p0_wins} / {args.pairs}")
    print(f"{res.agent0_name} 作为后手 (P1) 胜场: {res.agent0_as_p1_wins} / {args.pairs}")
    if res.reasons:
        print(
            f"🎯 终局胜因: 20声望胜 {res.reasons.get('20_points', 0)} 局 | "
            f"10皇冠胜 {res.reasons.get('10_crowns', 0)} 局 | "
            f"10单色胜 {res.reasons.get('10_color_points', 0)} 局"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()
