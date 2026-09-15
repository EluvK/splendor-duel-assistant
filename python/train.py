"""Main training launcher with sharded disk streaming and AlphaZero self-play."""

import argparse
import concurrent.futures
from pathlib import Path
import time
import torch

from splendor_ai._engine import evaluate_neural_match
from splendor_ai.advisor import HealthStatus, IterationRecord, TrainingAdvisor
from splendor_ai.dataset import FastTensorLoader, ReplayBuffer, ShardedBuffer
from splendor_ai.net import SplendorNet
from splendor_ai.selfplay import (
    generate_heuristic_compact_batch,
    generate_rust_neural_mcts_compact_batch,
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
    parser.add_argument("--games", type=int, default=20000, help="Total games to generate/train for imitation")
    parser.add_argument("--shard-games", type=int, default=2000, help="Games per shard (keeps memory bounded < 1GB)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--data-dir", type=str, default="data/shards", help="Directory to store sharded data")
    parser.add_argument("--reuse-data", action="store_true", help="Reuse existing shards in data-dir without re-generating")
    parser.add_argument("--clear-data", action="store_true", help="Clear data-dir before generating new shards")

    # 自博弈参数
    parser.add_argument("--iterations", type=int, default=30, help="Number of self-play iterations")
    parser.add_argument("--games-per-iter", type=int, default=50, help="Games to generate per self-play iteration")
    parser.add_argument("--mcts-sims", type=int, default=60, help="MCTS simulation count per move in self-play")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Max sample capacity for replay buffer")
    parser.add_argument("--temp-steps", type=int, default=36, help="Opening steps with temperature=1.0 + Dirichlet noise")
    parser.add_argument("--temp-final", type=float, default=0.25, help="Residual temperature after temp-steps to preserve mid-late game diversity")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha parameter")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25, help="Dirichlet noise weight")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument(
        "--eval-agent",
        type=str,
        choices=["neural_mcts", "policy_net"],
        default="neural_mcts",
        help="Evaluation agent type for promotion arena ('policy_net' for fast eval, 'neural_mcts' for deep eval)",
    )
    parser.add_argument("--train-epochs", type=int, default=3, help="Training epochs per iteration in self-play")
    parser.add_argument("--eval-pairs", type=int, default=30, help="Paired match count in arena evaluation (2 * pairs games)")
    parser.add_argument("--promote-threshold", type=float, default=0.56, help="Win-rate threshold to promote candidate to best")
    parser.add_argument(
        "--pipeline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable async double-buffering pipeline (overlap CPU MCTS self-play and GPU training)",
    )

    # 训练超参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (defaults: 8192 for imitation, 512 for selfplay)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (defaults: 1e-3 for imitation, 3e-4 for selfplay)",
    )
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
    if args.batch_size is None:
        args.batch_size = 8192
    if args.lr is None:
        args.lr = 1e-3
    print(
        f"⚙️  硬件设备: {device_str.upper()} | 目标局数: {args.games} 局 | 分片粒度: {args.shard_games} 局/分片 "
        f"| Batch: {args.batch_size} | 学习率: {args.lr} | 轮次: {args.epochs} Epochs"
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

    print("\n" + "=" * 90)
    print(
        f"{'Epoch':<8}{'Train Loss':<12}{'Policy':<10}{'Win Loss':<10}{'Turns Loss':<12}{'Top-1 Acc':<12}{'Val Loss':<10}"
    )
    print("=" * 90)

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
        win_loss_str = f"{train_metrics.get('win_loss', 0.0):.4f}"
        turns_loss_str = f"{train_metrics.get('turns_loss', 0.0):.4f}"
        print(
            f"{ep:<8}{train_metrics['loss']:<12.4f}{train_metrics['policy_loss']:<10.4f}"
            f"{win_loss_str:<10}{turns_loss_str:<12}"
            f"{top1_str:<12}"
            f"{val_metrics['eval_loss']:<10.4f}{'🌟 (Best)' if is_best else ''}"
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

    # 经验回放池 (滑动窗口防过拟合与遗忘)
    replay_buffer = ReplayBuffer(max_samples=args.buffer_size)

    # 智能诊断与自适应参数建议器 (默认全面监控 5 大异常场景并执行自适应早停)
    advisor = TrainingAdvisor()
    status: HealthStatus = HealthStatus.HEALTHY
    terminated_early = False

    if args.batch_size is None:
        args.batch_size = 512
    if args.lr is None:
        args.lr = 3e-4
    print(f"⚙️  自博弈训练超参: Batch Size = {args.batch_size} | 初始 LR = {args.lr}")

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

    # 异步双缓冲执行器 (派发后台 Rust Rayon 推演任务，实现 CPU 生成与 GPU 训练重叠)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _submit_selfplay_job(it_num: int, model_net: SplendorNet):
        seed = int(time.time()) + it_num * 1009
        t_start = time.time()
        onnx_bytes = model_net.export_onnx_bytes()
        fut = executor.submit(
            generate_rust_neural_mcts_compact_batch,
            None,
            onnx_bytes,
            args.games_per_iter,
            args.mcts_sims,
            seed,
            args.temp_steps,
            args.dirichlet_alpha,
            args.dirichlet_eps,
        )
        return fut, t_start

    active_pipeline = args.pipeline
    next_batch_fut = None
    next_batch_t0 = 0.0
    if active_pipeline:
        print("⚡ 启用异步双缓冲流水线 (CPU Rust MCTS 自对弈与 GPU 训练重叠并发)...")
        next_batch_fut, next_batch_t0 = _submit_selfplay_job(start_iter, baseline_net)

    # 2. 迭代飞轮
    end_iter = start_iter + args.iterations - 1
    for it in range(start_iter, end_iter + 1):
        print(f"\n" + "=" * 80)
        print(f"🔄 [AlphaZero 迭代轮次 {it}/{end_iter}]")
        print("=" * 80)

        # (A) 自对弈数据采样
        t_wait_start = time.time()
        if active_pipeline and next_batch_fut is not None:
            batch = next_batch_fut.result()
            wait_time = time.time() - t_wait_start
            total_gen_time = time.time() - next_batch_t0
            print(
                f"1. ✅ 自博弈数据就绪！新增 {batch.num_samples} 紧凑搜索样本 "
                f"(后台推演耗时: {total_gen_time:.2f}s | 主线程等待: {wait_time:.2f}s | "
                f"吞吐: {batch.num_samples/max(total_gen_time, 1e-6):.0f} 步/秒)"
            )
        else:
            t0 = time.time()
            print(
                f"1. 启动 Rust 8 线程并行 ONNX 纯神经网络 MCTS 自对弈 {args.games_per_iter} 局 "
                f"(推演: {args.mcts_sims} 次/步 | 前 {args.temp_steps} 步注入 Dirichlet 探索噪声与温度轮盘赌采样)..."
            )
            batch = generate_rust_neural_mcts_compact_batch(
                net=baseline_net,
                num_games=args.games_per_iter,
                num_simulations=args.mcts_sims,
                start_seed=int(time.time()) + it * 1009,
                temp_steps=args.temp_steps,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
            )
            gen_time = time.time() - t0
            print(
                f"   ✅ 本轮自博弈采样完成！新增 {batch.num_samples} 紧凑搜索样本 "
                f"(耗时: {gen_time:.2f}s | 吞吐: {batch.num_samples/max(gen_time, 1e-6):.0f} 步/秒)"
            )

        if batch.num_samples == 0:
            print("   ⚠️ 样本采集为空，跳过本轮训练。")
            continue

        replay_buffer.add_batch(batch)
        total_games += args.games_per_iter
        total_samples += batch.num_samples
        print(f"   📦 ReplayBuffer 经验池当前维护: {len(replay_buffer):,} 步有效样本")

        # 关键流水线动作：立刻在后台异步发射下一轮自对弈推演！
        if active_pipeline and it < end_iter:
            print("   🚀 [Pipeline] 后台异步预推演下一轮对局 (与 GPU 训练重叠并发)...")
            next_batch_fut, next_batch_t0 = _submit_selfplay_job(it + 1, baseline_net)
        else:
            next_batch_fut = None

        # (B) 候选模型拟合更新 (在滑动窗口缓冲池上训练)
        train_batch = replay_buffer.get_compact_batch()
        print(f"2. 训练候选模型 ({args.train_epochs} Epochs, 训练池规模: {train_batch.num_samples:,} 步)...")
        loader = FastTensorLoader(train_batch, batch_size=args.batch_size, shuffle=True, device=device)
        for ep in range(args.train_epochs):
            metrics = trainer.train_epoch(loader)
            if ep == args.train_epochs - 1:
                print(
                    f"   📉 拟合损失: Total {metrics['loss']:.4f} | Policy {metrics['policy_loss']:.4f} "
                    f"| Win {metrics.get('win_loss', 0.0):.4f} | Turns {metrics.get('turns_loss', 0.0):.4f} "
                    f"| Reason {metrics.get('reason_loss', 0.0):.4f} | Top-1: {metrics['top1_acc']*100:.1f}%"
                )

        # (C) 竞技场门禁对抗 (Candidate vs Baseline)
        eval_sims = args.mcts_sims if args.eval_agent == "neural_mcts" else 0
        eval_mode_desc = f"NeuralMCTS-{args.mcts_sims}" if eval_sims > 0 else "PolicyNet"
        print(f"3. 竞技场门禁对抗评测 ({args.eval_pairs * 2} 局成对严格换座对抗 | 决策: {eval_mode_desc})...")
        t_arena = time.time()
        bytes_c = candidate_net.export_onnx_bytes()
        bytes_b = baseline_net.export_onnx_bytes()
        total_g, c_wins, b_wins, draws, reasons = evaluate_neural_match(
            bytes_c,
            bytes_b,
            num_pairs=args.eval_pairs,
            base_seed=int(time.time()) + it * 503,
            num_sims=eval_sims,
        )
        win_rate = c_wins / max(total_g, 1)
        arena_elapsed = time.time() - t_arena
        print(
            f"   ⚔️ Rust 并发对决完成 (耗时 {arena_elapsed:.2f}s): 候选胜 {c_wins} 局 | 基准胜 {b_wins} 局 "
            f"| 平局 {draws} 局 | 候选胜率: {win_rate*100:.1f}%"
        )
        avg_rounds = 0.0
        avg_steps = 0.0
        if reasons:
            avg_rounds = reasons.get("total_rounds", 0) / max(total_g, 1)
            avg_steps = reasons.get("total_steps", 0) / max(total_g, 1)
            round_stats = [f"平均 {avg_rounds:.1f} 轮 ({avg_steps:.1f} 步)"]
            if c_wins > 0 and "agent0_win_rounds" in reasons:
                round_stats.append(f"候选胜均耗 {reasons['agent0_win_rounds'] / c_wins:.1f} 轮")
            if b_wins > 0 and "agent0_lose_rounds" in reasons:
                round_stats.append(f"基准胜均耗 {reasons['agent0_lose_rounds'] / b_wins:.1f} 轮")
            print(f"   ⏱️ 对局回合: {' | '.join(round_stats)}")
            print(
                f"   🎯 终局胜因: 20声望胜 {reasons.get('20_points', 0)} 局 | "
                f"10皇冠胜 {reasons.get('10_crowns', 0)} 局 | "
                f"10单色胜 {reasons.get('10_color_points', 0)} 局"
            )
        match_agent0_wins = c_wins
        match_agent1_wins = b_wins

        # 晋升判定
        promoted = win_rate >= args.promote_threshold
        meta = {
            "iteration": it,
            "total_games": total_games,
            "total_samples": total_samples,
            "win_rate": win_rate,
            "promoted": promoted,
            "candidate_wins": match_agent0_wins,
            "baseline_wins": match_agent1_wins,
            "avg_eval_rounds": avg_rounds,
            "avg_eval_steps": avg_steps,
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

        # (D) 智能诊断与自适应调优监控 (默认开启全场景异常早停与指标分析)
        record = IterationRecord(
            iteration=it,
            train_loss=metrics["loss"],
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            top1_acc=metrics["top1_acc"],
            top3_acc=metrics["top3_acc"],
            win_rate=win_rate,
            promoted=promoted,
            candidate_wins=match_agent0_wins,
            baseline_wins=match_agent1_wins,
            draws=draws,
            reasons=reasons or {},
            avg_rounds=avg_rounds,
            avg_steps=avg_steps,
            lr=metrics["lr"],
            samples_added=batch.num_samples,
            buffer_size=len(replay_buffer),
            current_args=vars(args),
        )
        status, decision, advice = advisor.step(record)
        print(advisor.format_step_summary(record, status, decision))

        if decision and decision.should_terminate:
            terminated_early = True
            if advice:
                print(advisor.format_terminal_report(decision, advice))
            if active_pipeline and next_batch_fut is not None:
                next_batch_fut.cancel()
                print("   🧹 正在安全回收后台异步自博弈推演资源...")
            break

    executor.shutdown(wait=False)
    if not terminated_early and advisor.history:
        last_rec = advisor.history[-1]
        final_advice = advisor.generate_advice(last_rec, status)
        print(advisor.format_terminal_report(None, final_advice))

    print(f"\n🏁 全部自博弈迭代完成！终局最强模型位于 {best_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "imitation":
        train_imitation(args)
    elif args.mode == "selfplay":
        train_selfplay(args)


if __name__ == "__main__":
    main()
