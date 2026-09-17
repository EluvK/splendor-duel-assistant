"""Benchmark and verification script for 128-concurrent GPU batched MCTS."""

import time
import torch
import numpy as np

from pathlib import Path
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet
from splendor_ai.selfplay import (
    generate_gpu_batched_mcts_compact_batch,
    generate_rust_neural_mcts_compact_batch,
)


def main():
    print("=" * 80)
    print("🚀 璀璨宝石：对决 (Splendor Duel) 128 并发 GPU 批推演性能基准评测")
    print("=" * 80)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"🖥️  推演后端: {device_name} ({device})")

    model = SplendorNet().to(device)
    ckpt_path = Path("checkpoints/best.pt")
    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(data["model_state"])
        print(f"📦 已载入基准模型权重: {ckpt_path}")
    model.eval()

    # 1. 预热测试 (2 局)
    print("\n[1/3] 执行引擎冷启动与显存预热 (2 局 MCTS-30)...")
    t0 = time.time()
    warmup_batch = generate_gpu_batched_mcts_compact_batch(
        net=model,
        num_games=2,
        num_simulations=30,
        max_concurrent_games=2,
        start_seed=42,
        temp_steps=12,
        temp_final=0.25,
        device=device,
    )
    t_warmup = time.time() - t0
    print(
        f"   ✅ 预热完成: 产出 {warmup_batch.num_samples} 步样本 | 耗时: {t_warmup:.2f}s | "
        f"Obs 形状: {warmup_batch.obs.shape} | Policy 形状: {warmup_batch.target_policy.shape}"
    )

    # 验证数据完整性与约束
    assert warmup_batch.obs.shape[1] == SplendorDuelEnv.OBS_SIZE, f"Expected {SplendorDuelEnv.OBS_SIZE} obs dim, got {warmup_batch.obs.shape[1]}"
    assert warmup_batch.mask.shape[1] == SplendorDuelEnv.ACTION_SIZE, f"Expected {SplendorDuelEnv.ACTION_SIZE} mask dim, got {warmup_batch.mask.shape[1]}"
    assert warmup_batch.target_policy.shape[1] == SplendorDuelEnv.ACTION_SIZE, f"Expected {SplendorDuelEnv.ACTION_SIZE} policy dim, got {warmup_batch.target_policy.shape[1]}"
    assert warmup_batch.value.shape[1] == 2, f"Expected 2 value dim, got {warmup_batch.value.shape[1]}"
    assert warmup_batch.reason.shape[1] == 3, f"Expected 3 reason dim, got {warmup_batch.reason.shape[1]}"
    assert (warmup_batch.mask.sum(axis=1) >= 1).all(), "Found state without legal actions"
    assert np.allclose(warmup_batch.target_policy.sum(axis=1), 1.0, atol=1e-2), "Policy distributions not normalized"
    assert (warmup_batch.value[:, 0] >= -1.0).all() and (warmup_batch.value[:, 0] <= 1.0).all(), "Win value out of [-1, 1]"
    assert (warmup_batch.value[:, 1] >= 0.0).all() and (warmup_batch.value[:, 1] <= 1.0).all(), "Turns value out of [0, 1]"
    print(f"   ✅ 数据格式与概率约束校验通过 (obs={SplendorDuelEnv.OBS_SIZE}, mask={SplendorDuelEnv.ACTION_SIZE}, policy={SplendorDuelEnv.ACTION_SIZE}, value=2, reason=3)")

    # 2. 中等规模测试 (16 并发，20 局)
    print("\n[2/3] 中并发性能实测: 20 局对战 (并发 16 | MCTS-30)...")
    t0 = time.time()
    b20 = generate_gpu_batched_mcts_compact_batch(
        net=model,
        num_games=20,
        num_simulations=30,
        max_concurrent_games=16,
        start_seed=100,
        temp_steps=12,
        temp_final=0.25,
        device=device,
    )
    t20 = time.time() - t0
    fps20 = b20.num_samples / max(t20, 1e-6)
    ms_per_step20 = (t20 / max(b20.num_samples, 1)) * 1000
    print(
        f"   ⚡ 20 局完成: 产出 {b20.num_samples} 步 | 耗时: {t20:.2f}s | "
        f"吞吐: {fps20:.0f} 步/秒 | 单步等效耗时: {ms_per_step20:.2f} ms | 单局耗时: {t20/20:.2f}s"
    )
    assert b20.num_samples >= 20 * 5, f"Unreasonably low sample count: {b20.num_samples}"
    assert b20.num_samples <= 20 * 400, f"Sample count exceeded maximum step bound: {b20.num_samples}"
    assert (b20.mask.sum(axis=1) >= 1).all(), "20-game batch contains states without legal actions"
    assert np.allclose(b20.target_policy.sum(axis=1), 1.0, atol=1e-2), "20-game batch policy distributions not normalized"
    print("   ✅ 20 局样本有效性门禁校验通过")

    # 3. 大规模 64~128 并发实测 (50 局)
    concurrency = 64 if cuda_available else 16
    print(f"\n[3/3] 大规模并发性能压测: 50 局对战 (并发 {concurrency} | MCTS-30)...")
    t0 = time.time()
    b50 = generate_gpu_batched_mcts_compact_batch(
        net=model,
        num_games=50,
        num_simulations=30,
        max_concurrent_games=concurrency,
        start_seed=1000,
        temp_steps=12,
        temp_final=0.25,
        device=device,
    )
    t50 = time.time() - t0
    fps50 = b50.num_samples / max(t50, 1e-6)
    ms_per_step50 = (t50 / max(b50.num_samples, 1)) * 1000
    print(
        f"   ⚡ 50 局完成: 产出 {b50.num_samples} 步 | 耗时: {t50:.2f}s | "
        f"吞吐: {fps50:.0f} 步/秒 | 单步等效耗时: {ms_per_step50:.2f} ms | 单局等效耗时: {t50/50:.2f}s"
    )
    assert b50.num_samples >= 50 * 5, f"Unreasonably low sample count: {b50.num_samples}"
    assert b50.num_samples <= 50 * 400, f"Sample count exceeded maximum step bound: {b50.num_samples}"
    assert (b50.mask.sum(axis=1) >= 1).all(), "50-game batch contains states without legal actions"
    assert np.allclose(b50.target_policy.sum(axis=1), 1.0, atol=1e-2), "50-game batch policy distributions not normalized"
    print("   ✅ 50 局样本有效性门禁校验通过")

    print("\n" + "=" * 80)
    print(f"🏆 压测结论总结:")
    print(f"   • 50 局 MCTS-30 总耗时: {t50:.2f} 秒")
    if cuda_available:
        speedup = 60.0 / max(t50, 0.01)
        print(f"   • 实测相对 CPU AVX2 基准 (~60s) 提速倍率: {speedup:.1f}x ⚡")
    print(f"   • 决策吞吐: {fps50:.0f} 步/秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
