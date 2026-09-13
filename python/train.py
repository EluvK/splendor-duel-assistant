"""Main training launcher with sharded disk streaming for Splendor Duel AI."""

import argparse
from pathlib import Path
import time
import torch

from splendor_ai.dataset import ShardedBuffer
from splendor_ai.net import SplendorNet
from splendor_ai.selfplay import (
    generate_heuristic_compact_batch,
    generate_selfplay_compact_batch,
)
from splendor_ai.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Splendor Duel AI Training Launcher (Sharded Streaming)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["imitation", "selfplay"],
        default="imitation",
        help="Training mode: 'imitation' or 'selfplay'",
    )
    # 模仿学习与分片参数
    parser.add_argument("--games", type=int, default=40000, help="Total games to generate/train")
    parser.add_argument("--shard-games", type=int, default=2500, help="Games per shard (keeps memory bounded < 1GB)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--data-dir", type=str, default="data/shards", help="Directory to store sharded data")
    parser.add_argument("--reuse-data", action="store_true", help="Reuse existing shards in data-dir without re-generating")
    parser.add_argument("--clear-data", action="store_true", help="Clear data-dir before generating new shards")

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for training")
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

    # 1. 检查是否需要生成分片
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
            del batch  # 立即释放单分片内存

        total_gen_time = time.time() - t_start
        print(f"✅ 全部分片落盘完成！总耗时: {total_gen_time:.2f}s | 目录: {args.data_dir}")

    buffer.refresh()
    all_shards = buffer.shard_files
    if not all_shards:
        print("❌ 未发现任何有效分片文件，训练终止。")
        return

    # 划分训练分片与验证分片 (最后一个分片作为专用验证分片)
    if len(all_shards) > 1:
        val_shard = all_shards[-1]
        train_shards = all_shards[:-1]
    else:
        val_shard = all_shards[0]
        train_shards = all_shards

    print(f"📊 分片划分: 训练分片 = {len(train_shards)} 个 | 验证分片 = {val_shard.name}")

    # 2. 初始化模型与训练器
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

    # 3. 流式分片训练循环
    print("\n" + "=" * 80)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Policy Loss':<14}{'Top-1 Acc':<14}{'Top-3 Acc':<14}{'Val Loss':<12}")
    print("=" * 80)

    best_val_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch_sharded(
            shard_files=train_shards, batch_size=args.batch_size, shuffle_shards=True
        )
        val_metrics = trainer.evaluate_sharded(val_shard, batch_size=args.batch_size)

        is_best = val_metrics["eval_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["eval_loss"]

        trainer.save_checkpoint("latest.pt", is_best=is_best, meta={"train": train_metrics, "val": val_metrics})

        top1_str = f"{train_metrics['top1_acc']*100:.1f}%"
        top3_str = f"{train_metrics['top3_acc']*100:.1f}%"
        print(
            f"{ep:<8}{train_metrics['loss']:<14.4f}{train_metrics['policy_loss']:<14.4f}"
            f"{top1_str:<14}{top3_str:<14}"
            f"{val_metrics['eval_loss']:<12.4f}{'🌟 (Best)' if is_best else ''}"
        )

    print("=" * 80)
    print(f"🎉 大规模分片训练完成！最优模型已保存至 {Path(args.ckpt_dir) / 'best.pt'}\n")


def main() -> None:
    args = parse_args()
    if args.mode == "imitation":
        train_imitation(args)


if __name__ == "__main__":
    main()
