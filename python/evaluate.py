"""Independent arena evaluation CLI tool for Splendor Duel."""

import argparse
from pathlib import Path
import time
from typing import Optional, Tuple
import torch

from splendor_ai.arena import Arena
from splendor_ai.mcts import (
    Agent,
    HeuristicAgent,
    MCTSAgent,
    PolicyNetAgent,
    RandomAgent,
    RustMCTSAgent,
)
from splendor_ai.net import SplendorNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Splendor Duel Arena Match Evaluator")
    parser.add_argument(
        "--agent1",
        type=str,
        choices=["rust_mcts", "mcts", "net", "heuristic", "random"],
        default="rust_mcts",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        choices=["rust_mcts", "mcts", "net", "heuristic", "random"],
        default="heuristic",
    )
    parser.add_argument("--model1", type=str, default="checkpoints/best.pt", help="Path to weights for agent 1")
    parser.add_argument("--model2", type=str, default=None, help="Path to weights for agent 2 (if mcts or net)")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulation count")
    parser.add_argument("--pairs", type=int, default=10, help="Paired match count (total games = 2 * pairs)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")

    return parser.parse_args()


def load_agent(agent_type: str, model_path: Optional[str], sims: int, device: torch.device) -> Tuple[Agent, str]:
    if agent_type == "rust_mcts":
        return RustMCTSAgent(num_sims=sims), f"RustMCTS-{sims}"
    elif agent_type == "heuristic":
        return HeuristicAgent(), "HeuristicAI"
    elif agent_type == "random":
        return RandomAgent(), "RandomAI"

    # 神经网络类型
    net = SplendorNet().to(device)
    name = f"Net({Path(model_path).name})" if model_path else "Net(Untrained)"

    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        net.load_state_dict(ckpt["model_state"])
        print(f"📦 成功加载模型: {model_path} (Epoch {ckpt.get('epoch', 0)})")
    else:
        print(f"⚠️ 权重文件 {model_path} 不存在，使用随机初始网络。")

    if agent_type == "mcts":
        return MCTSAgent(net, device, num_sims=sims, temperature=0.0), f"MCTS-{sims}({name})"
    else:
        return PolicyNetAgent(net, device), f"Policy({name})"


def main() -> None:
    args = parse_args()

    device_str = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    device = torch.device(device_str)

    print("============================================================")
    print("⚔️ 《璀璨宝石：对决》竞技场对抗评测 (Arena Match)")
    print(f"⚙️ 硬件设备: {device_str.upper()} | 成对局数: {args.pairs} (总共 {args.pairs * 2} 局)")
    print("============================================================")

    agent1, name1 = load_agent(args.agent1, args.model1, args.sims, device)
    agent2, name2 = load_agent(args.agent2, args.model2 or args.model1, args.sims, device)

    arena = Arena(agent1, agent2, agent0_name=name1, agent1_name=name2)
    t0 = time.time()
    res = arena.play_match(num_pairs=args.pairs, base_seed=int(time.time()))
    dur = time.time() - t0

    print("\n================== 比赛结果战报 ==================")
    print(f"耗时: {dur:.1f} 秒 | 总局数: {res.total_games} 局 | 平均步数: {res.avg_steps:.1f}")
    print("-" * 50)
    print(f"🥇 {res.agent0_name:<25} 胜场: {res.agent0_wins:>3} 局 ({res.agent0_win_rate*100:5.1f}%)")
    print(f"🥈 {res.agent1_name:<25} 胜场: {res.agent1_wins:>3} 局 ({(1.0 - res.agent0_win_rate)*100:5.1f}%)")
    if res.draws > 0:
        print(f"🤝 平局: {res.draws} 局")
    print("-" * 50)
    print(f"{res.agent0_name} 作为先手 (P0) 胜场: {res.agent0_as_p0_wins} / {args.pairs}")
    print(f"{res.agent0_name} 作为后手 (P1) 胜场: {res.agent0_as_p1_wins} / {args.pairs}")
    print("==================================================")


if __name__ == "__main__":
    main()
