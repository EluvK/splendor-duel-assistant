"""Main training launcher for Splendor Duel AI."""

import argparse
from pathlib import Path
import random
import sys
import time
import torch
from torch.utils.data import DataLoader, random_split

from splendor_ai.dataset import ReplayBuffer, SplendorDataset
from splendor_ai.net import SplendorNet
from splendor_ai.selfplay import generate_heuristic_dataset, generate_selfplay_dataset
from splendor_ai.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Splendor Duel AI Training Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["imitation", "selfplay"],
        default="imitation",
        help="Training mode: 'imitation' (behavioral cloning from heuristic AI) or 'selfplay' (AlphaZero loop)",
    )
    # 模仿学习参数
    parser.add_argument("--games", type=int, default=1000, help="Number of games to generate for imitation learning")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for imitation learning")

    # 自博弈参数
    parser.add_argument("--iterations", type=int, default=10, help="Number of self-play iterations")
    parser.add_argument("--games-per-iter", type=int, default=100, help="Games to generate per self-play iteration")
    parser.add_argument("--train-epochs", type=int, default=3, help="Training epochs per iteration in self-play")
    parser.add_argument("--buffer-capacity", type=int, default=100_000, help="Max capacity of replay buffer")

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")

    # 路径与恢复
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")

    return parser.parse_args()


def train_imitation(args: argparse.Namespace) -> None:
    print(f"\n🚀 启动模仿学习 (Imitation Learning / 行为克隆冷启动)...")
    device_str = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    print(f"⚙️  硬件设备: {device_str.upper()} | 目标局数: {args.games} 局 | 轮次: {args.epochs} Epochs")

    # 1. 采集数据
    t0 = time.time()
    print(f"正在从 Rust 启发式 AI 生成 {args.games} 局专家对战数据...")
    samples = generate_heuristic_dataset(num_games=args.games, start_seed=int(time.time()))
    gen_time = time.time() - t0
    print(f"✅ 数据采样完成！共生成 {len(samples)} 个决策步样本 (耗时: {gen_time:.2f}s, 约 {len(samples)/gen_time:.0f} 步/秒)")

    # 2. 划分训练集与验证集 (9:1)
    dataset = SplendorDataset(samples)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 3. 初始化模型与训练器
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
        meta = trainer.load_checkpoint(Path(args.resume))
        print(f"🔄 从检查点 {args.resume} 恢复状态 (Epoch {trainer.epoch})")

    # 4. 训练循环
    print("\n" + "=" * 80)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Policy Loss':<14}{'Top-1 Acc':<14}{'Top-3 Acc':<14}{'Val Loss':<12}")
    print("=" * 80)

    best_val_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

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
    print(f"🎉 模仿学习完成！最优检查点已保存至 {Path(args.ckpt_dir) / 'best.pt'}\n")


def train_selfplay(args: argparse.Namespace) -> None:
    print(f"\n🚀 启动自博弈强化学习 (AlphaZero Self-Play Loop)...")
    device_str = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    device = torch.device(device_str)
    print(f"⚙️  硬件设备: {device_str.upper()} | 迭代次数: {args.iterations} | 每次迭代对局: {args.games_per_iter} 局")

    net = SplendorNet()
    cfg = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=device_str,
        amp=not args.no_amp,
        ckpt_dir=args.ckpt_dir,
        t_max_epochs=args.iterations * args.train_epochs,
    )
    trainer = Trainer(net, cfg)

    if args.resume:
        meta = trainer.load_checkpoint(Path(args.resume))
        print(f"🔄 从检查点 {args.resume} 恢复")

    buffer = ReplayBuffer(max_samples=args.buffer_capacity)

    for it in range(1, args.iterations + 1):
        print(f"\n--- [Iteration {it}/{args.iterations}] ---")
        t0 = time.time()
        print(f"使用当前网络自对弈采样 {args.games_per_iter} 局...")
        new_samples = generate_selfplay_dataset(
            trainer.net, device, num_games=args.games_per_iter, start_seed=int(time.time()) + it * 1000
        )
        buffer.add_samples(new_samples)
        gen_time = time.time() - t0
        print(f"新增 {len(new_samples)} 样本 (耗时: {gen_time:.1f}s) | 经验池当前样本总量: {len(buffer)}")

        # 训练更新当前网络
        dataset = buffer.to_dataset()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        for ep in range(args.train_epochs):
            metrics = trainer.train_epoch(loader)
            print(
                f"  Iter {it} Epoch {ep+1}/{args.train_epochs}: Loss = {metrics['loss']:.4f} "
                f"| Policy Loss = {metrics['policy_loss']:.4f} | Value Loss = {metrics['value_loss']:.4f} "
                f"| Top-1 Acc = {metrics['top1_acc']*100:.1f}%"
            )

        trainer.save_checkpoint("latest.pt", is_best=True, meta={"iteration": it, "samples": len(buffer)})

    print(f"\n🎉 自博弈迭代完成！最新权重保存在 {Path(args.ckpt_dir) / 'latest.pt'}")


def main() -> None:
    args = parse_args()
    if args.mode == "imitation":
        train_imitation(args)
    elif args.mode == "selfplay":
        train_selfplay(args)


if __name__ == "__main__":
    main()
