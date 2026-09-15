"""Comprehensive performance diagnosis and benchmarking script for Splendor Duel."""

import io
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

from splendor_ai.net import SplendorNet
from splendor_ai.dataset import CompactBatch, FastTensorLoader, ShardedBuffer, ReplayBuffer
from splendor_ai.trainer import Trainer, TrainerConfig
from splendor_ai.selfplay import (
    generate_heuristic_compact_batch,
    generate_rust_neural_mcts_compact_batch,
    generate_league_mcts_compact_batch,
)
from splendor_ai._engine import (
    evaluate_neural_match,
    PyGameState,
)

def format_speed(items, duration, unit="items"):
    rate = items / max(duration, 1e-6)
    return f"{rate:,.1f} {unit}/s (耗时 {duration:.3f}s)"

def test_heuristic_generation(num_games=1000):
    print("\n" + "=" * 70)
    print(f"📊 [测试 1] 启发式轨迹采样与跨语言桥接性能 (采样 {num_games} 局)")
    print("=" * 70)
    
    t0 = time.perf_counter()
    batch = generate_heuristic_compact_batch(num_games=num_games, start_seed=12345)
    t1 = time.perf_counter()
    duration = t1 - t0
    
    total_steps = batch.num_samples
    print(f"  • 生成局数: {num_games} 局")
    print(f"  • 采样总步数: {total_steps:,} 步 (平均每局 {total_steps/num_games:.1f} 步)")
    print(f"  • 端到端总耗时: {duration:.3f} 秒")
    print(f"  • 对局吞吐率: {num_games / duration:,.1f} 局/秒")
    print(f"  • 决策步吞吐率: {total_steps / duration:,.1f} 步/秒")
    print(f"  • 紧凑数组内存: obs={batch.obs.nbytes / 1024**2:.1f}MB, policy={batch.target_policy.nbytes / 1024**2:.1f}MB, mask={batch.mask.nbytes / 1024**2:.1f}MB")
    
    return batch

def test_sharded_io(batch, shard_dir="data/bench_shards"):
    print("\n" + "=" * 70)
    print("📊 [测试 2] 磁盘分片 I/O 读写吞吐性能")
    print("=" * 70)
    p = Path(shard_dir)
    p.mkdir(parents=True, exist_ok=True)
    shard_file = p / "bench_shard.npz"
    
    # 1. 未压缩保存
    t0 = time.perf_counter()
    batch.save_npz(shard_file, compressed=False)
    t1 = time.perf_counter()
    file_size_mb = shard_file.stat().st_size / 1024**2
    save_duration = t1 - t0
    print(f"  • np.savez (未压缩): 大小={file_size_mb:.1f}MB, 耗时={save_duration:.3f}s, 写入带宽={file_size_mb/save_duration:.1f} MB/s")
    
    # 2. 未压缩加载
    t0 = time.perf_counter()
    loaded_batch = CompactBatch.load_npz(shard_file)
    t1 = time.perf_counter()
    load_duration = t1 - t0
    print(f"  • np.load (未压缩): 样本数={loaded_batch.num_samples:,}, 耗时={load_duration:.3f}s, 读取带宽={file_size_mb/load_duration:.1f} MB/s")
    
    # 3. 压缩保存测试
    t0 = time.perf_counter()
    batch.save_npz(shard_file, compressed=True)
    t1 = time.perf_counter()
    comp_size_mb = shard_file.stat().st_size / 1024**2
    comp_save_duration = t1 - t0
    print(f"  • np.savez_compressed (压缩): 大小={comp_size_mb:.1f}MB (压缩比 {file_size_mb/comp_size_mb:.2f}x), 耗时={comp_save_duration:.3f}s")
    
    # 清理
    if shard_file.exists():
        shard_file.unlink()

def test_training_throughput(batch, device_str="cuda"):
    print("\n" + "=" * 70)
    print(f"📊 [测试 3] 神经网络训练前向/反向吞吐与 Batch Size 扩展性 ({device_str.upper()})")
    print("=" * 70)
    device = torch.device(device_str)
    net = SplendorNet().to(device)
    
    batch_sizes = [256, 512, 1024, 2048, 4096, 8192]
    
    for bs in batch_sizes:
        if bs > batch.num_samples:
            continue
        cfg = TrainerConfig(batch_size=bs, device=device_str, amp=True)
        trainer = Trainer(net, cfg)
        
        # 测 FastTensorLoader 构造时间
        t0 = time.perf_counter()
        loader = FastTensorLoader(batch, batch_size=bs, shuffle=True, device=device)
        loader_prep_time = time.perf_counter() - t0
        
        # 预热 1 次
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时 1 个 epoch
        t0 = time.perf_counter()
        metrics = trainer.train_epoch(loader)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0
        
        throughput = batch.num_samples / train_time
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        print(f"  • Batch={bs:<5} | 训练耗时: {train_time:.3f}s | 吞吐: {throughput:,.0f} samples/s | Loader耗时: {loader_prep_time*1000:.1f}ms | 显存峰值: {vram_mb:.0f}MB")

def test_onnx_and_evaluator_overhead():
    print("\n" + "=" * 70)
    print("📊 [测试 4] ONNX 导出、解析与图优化耗时")
    print("=" * 70)
    net = SplendorNet()
    
    # 测量 export_onnx_bytes
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        onnx_bytes = net.export_onnx_bytes()
        times.append(time.perf_counter() - t0)
    avg_export_ms = np.mean(times) * 1000
    print(f"  • PyTorch torch.onnx.export: 平均耗时 {avg_export_ms:.1f} ms (大小: {len(onnx_bytes)/1024:.1f} KB)")
    
    # 测量 evaluate_neural_match 初始化的开销 (通过对抗 0 局测初始化)
    t0 = time.perf_counter()
    # 换座对抗评测 1 对对决 (2 局)，纯 PolicyNet (num_sims=0)
    total_g, c_wins, b_wins, draws, _ = evaluate_neural_match(onnx_bytes, onnx_bytes, num_pairs=1, base_seed=42, num_sims=0)
    t1 = time.perf_counter()
    print(f"  • Rust tract-onnx 载入/优化 + 2 局 PolicyNet 极速对决: 耗时 {(t1 - t0)*1000:.1f} ms")

def test_neural_mcts_selfplay(onnx_bytes):
    print("\n" + "=" * 70)
    print("📊 [测试 5] Neural MCTS 自博弈推演性能 (Rust 8 线程并行)")
    print("=" * 70)
    
    for num_sims in [0, 15, 30, 60]:
        t0 = time.perf_counter()
        if num_sims == 0:
            # PolicyNet 模式 (在 evaluate_neural_match 中测试)
            total_g, c_wins, b_wins, draws, _ = evaluate_neural_match(onnx_bytes, None, num_pairs=5, base_seed=42, num_sims=0)
            duration = time.perf_counter() - t0
            print(f"  • PolicyNet 极速直觉决策 (10 局对抗 HeuristicAI): 耗时 {duration:.2f}s (吞吐: {10/duration:.1f} 局/秒)")
        else:
            games = 4
            t0 = time.perf_counter()
            batch = generate_rust_neural_mcts_compact_batch(
                onnx_bytes=onnx_bytes,
                num_games=games,
                num_simulations=num_sims,
                start_seed=100,
                temp_steps=12,
            )
            duration = time.perf_counter() - t0
            steps = batch.num_samples
            gps = games / duration
            sps = steps / duration
            ms_per_step = (duration / steps) * 1000 * 8  # 单线程等效毫秒
            print(f"  • Neural MCTS (sims={num_sims:<2}, {games} 局自对弈): 耗时 {duration:.2f}s | {gps:.2f} 局/秒 | {sps:.1f} 步/秒 (单核每步决策约 {ms_per_step:.1f} ms)")

def test_full_iteration_profile(batch, onnx_bytes):
    print("\n" + "=" * 70)
    print("📊 [测试 6] 典型 AlphaZero 单轮 Iteration 端到端各阶段耗时比例模拟")
    print("=" * 70)
    
    print("正在测量标准一轮迭代 (50 局自博弈 + 3 Epochs 训练 + 60 局门禁评测) 的时间消耗模型...")
    
    # 测 2 局的 MCTS-30 时间，外推 50 局
    t0 = time.perf_counter()
    b2 = generate_rust_neural_mcts_compact_batch(onnx_bytes=onnx_bytes, num_games=2, num_simulations=30, start_seed=999)
    dur_2g = time.perf_counter() - t0
    extrap_selfplay_50g = (dur_2g / 2.0) * 50.0
    
    # 测 3 epochs 训练 10000 样本
    small_obs = batch.obs[:10000]
    small_batch = CompactBatch(
        obs=small_obs,
        mask=batch.mask[:10000],
        target_policy=batch.target_policy[:10000],
        value=batch.value[:10000],
        reason=batch.reason[:10000],
    )
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    net = SplendorNet().to(device_str)
    trainer = Trainer(net, TrainerConfig(batch_size=512, device=device_str))
    loader = FastTensorLoader(small_batch, batch_size=512, shuffle=True, device=torch.device(device_str))
    
    t0 = time.perf_counter()
    for _ in range(3):
        trainer.train_epoch(loader)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dur_train = time.perf_counter() - t0
    
    # 测 2 pairs (4 局) 的 evaluate_neural_match 时间，外推 30 pairs (60 局)
    # 分别测 PolicyNet (sims=0) 和 NeuralMCTS (sims=30)
    t0 = time.perf_counter()
    evaluate_neural_match(onnx_bytes, onnx_bytes, num_pairs=2, base_seed=42, num_sims=0)
    dur_eval_policynet = (time.perf_counter() - t0) / 4.0 * 60.0
    
    t0 = time.perf_counter()
    evaluate_neural_match(onnx_bytes, onnx_bytes, num_pairs=1, base_seed=42, num_sims=30)
    dur_eval_mcts30 = (time.perf_counter() - t0) / 2.0 * 60.0
    
    print("\n--- [单轮迭代耗时分析 (sims=30 方案)] ---")
    print(f"  [1] 自对弈采样 (50 局 MCTS-30 并发):         {extrap_selfplay_50g:>6.1f} s  ({extrap_selfplay_50g/(extrap_selfplay_50g+dur_train+dur_eval_mcts30)*100:.1f}%)")
    print(f"  [2] 候选网络拟合训练 (3 Epochs, 10k 样本):   {dur_train:>6.1f} s  ({dur_train/(extrap_selfplay_50g+dur_train+dur_eval_mcts30)*100:.1f}%)")
    print(f"  [3] 竞技场门禁对抗 (60 局 MCTS-30 评测):      {dur_eval_mcts30:>6.1f} s  ({dur_eval_mcts30/(extrap_selfplay_50g+dur_train+dur_eval_mcts30)*100:.1f}%)")
    total_mcts = extrap_selfplay_50g + dur_train + dur_eval_mcts30
    print(f"  => 单轮总耗时 (MCTS 门禁):                    {total_mcts:>6.1f} s (~{total_mcts/60:.1f} 分钟/迭代)")
    
    print("\n--- [单轮迭代耗时分析 (PolicyNet 快速门禁方案)] ---")
    total_fast = extrap_selfplay_50g + dur_train + dur_eval_policynet
    print(f"  [1] 自对弈采样 (50 局 MCTS-30 并发):         {extrap_selfplay_50g:>6.1f} s  ({extrap_selfplay_50g/total_fast*100:.1f}%)")
    print(f"  [2] 候选网络拟合训练 (3 Epochs, 10k 样本):   {dur_train:>6.1f} s  ({dur_train/total_fast*100:.1f}%)")
    print(f"  [3] 竞技场门禁对抗 (60 局 PolicyNet 极速):   {dur_eval_policynet:>6.1f} s  ({dur_eval_policynet/total_fast*100:.1f}%)")
    print(f"  => 单轮总耗时 (PolicyNet 极速门禁):           {total_fast:>6.1f} s (~{total_fast/60:.1f} 分钟/迭代)")

def main():
    print("🔍 启动《璀璨宝石：对决》全流程端到端性能诊断套件...")
    batch = test_heuristic_generation(num_games=500)
    test_sharded_io(batch)
    if torch.cuda.is_available():
        test_training_throughput(batch, device_str="cuda")
    # test_training_throughput(batch, device_str="cpu")
    
    test_onnx_and_evaluator_overhead()
    net = SplendorNet()
    onnx_bytes = net.export_onnx_bytes()
    test_neural_mcts_selfplay(onnx_bytes)
    test_full_iteration_profile(batch, onnx_bytes)
    print("\n✅ 全流程性能测试完成！")

if __name__ == "__main__":
    main()
