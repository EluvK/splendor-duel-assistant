"""Main training launcher with sharded disk streaming and AlphaZero self-play."""

import argparse
from pathlib import Path
import time
import torch

from splendor_ai.arena import Arena
from splendor_ai.dataset import FastTensorLoader, ShardedBuffer
from splendor_ai.mcts import MCTSAgent, PolicyNetAgent
from splendor_ai.net import SplendorNet
from splendor_ai.selfplay import (
    generate_heuristic_compact_batch,
    generate_mcts_selfplay_compact_batch,
)
from splendor_ai.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Splendor Duel AI Training Launcher (Sharded & Self-Play)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["imitation", "selfplay"],
        default="imitation",
        help="Training mode: 'imitation' (bootstrap from heuristic AI) or 'selfplay' (AlphaZero loop)",
    )
    # 模仿学习参数
    parser.add_argument("--games", type=int, default=10000, help="Total games to generate/train for imitation")
    parser.add_argument("--shard-games", type=int, default=2500, help="Games per shard (keeps memory bounded < 1GB)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--data-dir", type=str, default="data/shards", help="Directory to store sharded data")
    parser.add_argument("--reuse-data", action="store_true", help="Reuse existing shards in data-dir without re-generating")
    parser.add_argument("--clear-data", action="store_true", help="Clear data-dir before generating new shards")

    # 自博弈参数
    parser.add_argument("--iterations", type=int, default=10, help="Number of self-play iterations")
    parser.add_argument("--games-per-iter", type=int, default=100, help="Games to generate per self-play iteration")
    parser.add_argument("--mcts-sims", type=int, default=30, help="MCTS simulation count per move in self-play")
    parser.add_argument("--train-epochs", type=int, default=3, help="Training epochs per iteration in self-play")
    parser.add_argument("--eval-pairs", type=int, default=5, help="Paired match count in arena evaluation (2 * pairs games)")
    parser.add_argument("--promote-threshold", type=float, default=0.55, help="Win-rate threshold to promote candidate to best")

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")

    # 检查点
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")

    return parser.parse_args()


def train_imitation(args: argparse.Namespace) -> None:
    print(f"\n🚀 启动大规模分片模仿学习 (Sharded Imitation Learning)...")
    device_str = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    print(
        f"⚙️  硬件设备: {device_str.upper()} | 目标局数: {args.games} 局 | 分片粒度: {args.shard_games} 局/分片 "
        f"| Batch: {args.batch_size} | 轮次: {args.epochs} Epochs"
    )

    buffer = ShardedBuffer(Path(args.data_dir))
    if args.clear_data:
        print(f"🧹 清理历史分片目录: {args.data_dir}")
        buffer.clear()

    buffer.refresh()
    if args.reuse_data and len(buffer.shard_files) > 0:
        print(f"📦 发现现有分片 {len(buffer.shard_files)} 个，直接复用磁盘数据！")
    else:
        if len(buffer.shard_files) > 0 and not args.reuse_data:
            print(f"🧹 重新采样，清理已有分片 {len(buffer.shard_files)} 个...")
            buffer.clear()

        num_shards = (args.games + args.shard_games - 1) // args.shard_games
        print(f"开始分批全核并发生成 {num_shards} 个磁盘分片 (每分片 {args.shard_games} 局，内存严格受控)...")

        t_start = time.time()
        for i in range(num_shards):
            t0 = time.time()
            seed = int(time.time()) + i * 10007
            games_this_shard = min(args.shard_games, args.games - i * args.shard_games)

            batch = generate_heuristic_compact_batch(num_games=games_this_shard, start_seed=seed)
            shard_path = buffer.add_shard(batch, compressed=False)
            gen_time = time.time() - t0

            print(
                f"  [{i+1}/{num_shards}] 写入 {shard_path.name} | 样本: {batch.num_samples} 步 "
                f"(耗时: {gen_time:.2f}s | 吞吐: {batch.num_samples/gen_time:.0f} 步/秒)"
            )
            del batch

        total_gen_time = time.time() - t_start
        print(f"✅ 全部分片落盘完成！总耗时: {total_gen_time:.2f}s | 目录: {args.data_dir}")

    buffer.refresh()
    all_shards = buffer.shard_files
    if not all_shards:
        print("❌ 未发现任何有效分片文件，训练终止。")
        return

    if len(all_shards) > 1:
        val_shard = all_shards[-1]
        train_shards = all_shards[:-1]
    else:
        val_shard = all_shards[0]
        train_shards = all_shards

    print(f"📊 分片划分: 训练分片 = {len(train_shards)} 个 | 验证分片 = {val_shard.name}")
    total_samples = buffer.count_total_samples()

    net = SplendorNet()
    cfg = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=device_str,
        amp=not args.no_amp,
        ckpt_dir=args.ckpt_dir,
        t_max_epochs=args.epochs,
    )
    trainer = Trainer(net, cfg)

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        print(f"🔄 从检查点 {args.resume} 恢复")

    print("\n" + "=" * 80)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Policy Loss':<14}{'Top-1 Acc':<14}{'Top-3 Acc':<14}{'Val Loss':<12}")
    print("=" * 80)

    best_val_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch_sharded(
            shard_files=train_shards, batch_size=args.batch_size, shuffle_shards=True
        )
        val_metrics = trainer.evaluate_sharded(val_shard, batch_size=args.batch_size)

        meta = {
            "epoch": ep,
            "total_games": args.games,
            "total_samples": total_samples,
            "train": train_metrics,
            "val": val_metrics,
        }
        is_best = val_metrics["eval_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["eval_loss"]
            trainer.save_checkpoint(f"epoch_{ep:03d}.pt", is_best=True, meta=meta)
        trainer.save_checkpoint("latest.pt", is_best=False, meta=meta)

        top1_str = f"{train_metrics['top1_acc']*100:.1f}%"
        top3_str = f"{train_metrics['top3_acc']*100:.1f}%"
        print(
            f"{ep:<8}{train_metrics['loss']:<14.4f}{train_metrics['policy_loss']:<14.4f}"
            f"{top1_str:<14}{top3_str:<14}"
            f"{val_metrics['eval_loss']:<12.4f}{'🌟 (Best)' if is_best else ''}"
        )

    print("=" * 80)
    print(f"🎉 模仿学习完成！最优模型已保存至 {Path(args.ckpt_dir) / 'best.pt'}\n")


def train_selfplay(args: argparse.Namespace) -> None:
    print(f"\n🚀 启动 AlphaZero 自博弈强化学习飞轮 (MCTS Self-Play Loop)...")
    device_str = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    device = torch.device(device_str)
    print(
        f"⚙️  硬件: {device_str.upper()} | 迭代: {args.iterations} 轮 | 每轮自对弈: {args.games_per_iter} 局 "
        f"| MCTS 推演: {args.mcts_sims} 次/步 | 晋升门禁: {args.promote_threshold*100:.0f}%"
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    # 1. 初始化基准模型 (若指定 resume 或已有 best.pt 则热启)
    baseline_net = SplendorNet().to(device)
    start_iter = 1
    last_epoch = 0
    total_games = 0
    total_samples = 0
    ckpt_loaded = None

    if args.resume and Path(args.resume).exists():
        ckpt_loaded = torch.load(args.resume, map_location=device)
        print(f"🔄 从指定检查点恢复基准模型: {args.resume}")
    elif best_path.exists():
        ckpt_loaded = torch.load(best_path, map_location=device)
        print(f"🏆 成功加载现有基准冠军模型: {best_path}")
    else:
        print("🌱 未发现已存模型，从随机初始网络开始自对弈...")

    if ckpt_loaded is not None:
        baseline_net.load_state_dict(ckpt_loaded["model_state"])
        last_meta = ckpt_loaded.get("meta", {})
        last_it = last_meta.get("iteration", 0)
        start_iter = last_it + 1
        last_epoch = ckpt_loaded.get("epoch", 0)
        total_games = last_meta.get("total_games", last_it * args.games_per_iter)
        total_samples = last_meta.get("total_samples", 0)
        print(
            f"   📊 继承历史档案: 已完成迭代 {last_it} 轮 | 累计 Epoch: {last_epoch} | "
            f"累计对局: {total_games:,} 局 | 累计样本: {total_samples:,} 步"
        )

    # 候选训练模型
    candidate_net = SplendorNet().to(device)
    candidate_net.load_state_dict(baseline_net.state_dict())

    cfg = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=device_str,
        amp=not args.no_amp,
        ckpt_dir=args.ckpt_dir,
        t_max_epochs=args.iterations * args.train_epochs,
    )
    trainer = Trainer(candidate_net, cfg)
    trainer.epoch = last_epoch

    # 2. 迭代飞轮
    end_iter = start_iter + args.iterations - 1
    for it in range(start_iter, end_iter + 1):
        print(f"\n" + "=" * 80)
        print(f"🔄 [AlphaZero 迭代轮次 {it}/{end_iter}]")
        print("=" * 80)

        # (A) MCTS 深度推演自对弈采样 (Rust 8 线程并发)
        t0 = time.time()
        print(f"1. 正在由 Rust 8 线程并行驱动 MCTS 深度推演自对弈 {args.games_per_iter} 局 (每次决策 {args.mcts_sims} 次推演)...")
        batch = generate_mcts_selfplay_compact_batch(
            num_games=args.games_per_iter,
            num_simulations=args.mcts_sims,
            start_seed=int(time.time()) + it * 1009,
        )
        gen_time = time.time() - t0
        print(f"   ✅ 自博弈采样完成！共生成 {batch.num_samples} 紧凑搜索样本 (耗时: {gen_time:.2f}s | 吞吐: {batch.num_samples/max(gen_time, 1e-6):.0f} 步/秒)")

        if batch.num_samples == 0:
            print("   ⚠️ 样本采集为空，跳过本轮训练。")
            continue

        total_games += args.games_per_iter
        total_samples += batch.num_samples

        # (B) 候选模型拟合更新
        print(f"2. 训练候选模型 ({args.train_epochs} Epochs)...")
        loader = FastTensorLoader(batch, batch_size=args.batch_size, shuffle=True, device=device)
        for ep in range(args.train_epochs):
            metrics = trainer.train_epoch(loader)

        # (C) 竞技场门禁对抗 (Candidate vs Baseline)
        print(f"3. 竞技场门禁对抗评测 ({args.eval_pairs * 2} 局成对对抗)...")
        candidate_agent = PolicyNetAgent(candidate_net, device)
        baseline_agent = PolicyNetAgent(baseline_net, device)

        arena = Arena(
            candidate_agent,
            baseline_agent,
            agent0_name=f"Candidate-Iter{it}",
            agent1_name="Baseline-Best",
        )
        match_result = arena.play_match(num_pairs=args.eval_pairs, base_seed=int(time.time()) + it * 503)

        win_rate = match_result.agent0_win_rate
        print(
            f"   ⚔️ 对决结果: 候选胜 {match_result.agent0_wins} 局 | 基准胜 {match_result.agent1_wins} 局 "
            f"| 平局 {match_result.draws} 局 | 候选胜率: {win_rate*100:.1f}%"
        )

        # 晋升判定
        promoted = win_rate >= args.promote_threshold
        meta = {
            "iteration": it,
            "total_games": total_games,
            "total_samples": total_samples,
            "win_rate": win_rate,
            "promoted": promoted,
            "candidate_wins": match_result.agent0_wins,
            "baseline_wins": match_result.agent1_wins,
        }

        if promoted:
            print(f"   🎉 胜率达到 {win_rate*100:.1f}% (>= {args.promote_threshold*100:.0f}%) -> 晋升为新主力！🌟")
            baseline_net.load_state_dict(candidate_net.state_dict())
            iter_filename = f"iter_{it:03d}.pt"
            trainer.save_checkpoint(iter_filename, is_best=True, meta=meta)
            print(f"   💾 成功归档晋升存档: {ckpt_dir / iter_filename} 并同步更新主力 {best_path}")
        else:
            print(f"   ⚠️ 胜率 {win_rate*100:.1f}% 未达门禁要求 ({args.promote_threshold*100:.0f}%) -> 淘汰放弃，保留原基准重新探索。")
            candidate_net.load_state_dict(baseline_net.state_dict())

        trainer.save_checkpoint("latest.pt", is_best=False, meta=meta)

    print(f"\n🏁 全部自博弈迭代完成！终局最强模型位于 {best_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "imitation":
        train_imitation(args)
    elif args.mode == "selfplay":
        train_selfplay(args)


if __name__ == "__main__":
    main()
