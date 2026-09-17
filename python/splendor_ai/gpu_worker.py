"""High-performance GPU batched neural evaluator for Splendor Duel AlphaZero MCTS.

Provides vectorized tensor batching and PyTorch CUDA AMP FP16 accelerated batch forward passes.
"""

from typing import Tuple, Union
import numpy as np
import torch

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


class GpuBatchedEvaluator:
    """PyTorch GPU 批处理神经网络评估器.

    专为底层 Rust BatchedMctsRunner 设计：
    1. 接收来自 Rust 的 flat numpy 缓冲区，通过 torch.from_numpy 映射后推入 GPU 显存；
    2. 开启 PyTorch CUDA AMP FP16 半精度加速，吃满 GPU Tensor Core 算力；
    3. 前向计算完成后将多目标预测结果切片展平为 contiguous float32 numpy 数组传回 Rust。
    """

    def __init__(
        self,
        model: SplendorNet,
        device: Union[torch.device, str] = "cuda",
        warmup_batch_size: int = 128,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(
                "cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu"
            )
        self.device = device
        self.is_cuda = self.device.type == "cuda"

        self.model = model.to(self.device).eval()

        # 显存预热与 CUDA JIT / 驱动初始化
        with torch.no_grad():
            dummy_obs = torch.zeros(
                warmup_batch_size, SplendorDuelEnv.OBS_SIZE, dtype=torch.float32, device=self.device
            )
            if self.is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    self.model(dummy_obs)
                torch.cuda.synchronize(self.device)
            else:
                self.model(dummy_obs)

    def __call__(self, flat_obs: np.ndarray, count: int) -> np.ndarray:
        """批量前向推理回调 (由 Rust PyO3 桥接层直接调用).

        Args:
            flat_obs: 形状为 (count * 969,) 的 1D 连续 float32 numpy 数组
            count: 当前批次包含的有效盘面数量 (1 ~ 128)

        Returns:
            fused_np: (count * 1861,) 融合输出展平数组 [Logits(1856) + Win(1) + Turns(1) + Reason(3)]
        """
        obs_tensor = torch.from_numpy(flat_obs).view(count, SplendorDuelEnv.OBS_SIZE).to(
            self.device, non_blocking=True
        )

        with torch.no_grad():
            if self.is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    policy_logits, win_value, turns_value, reason_logits = self.model(obs_tensor)
                    reason_probs = torch.sigmoid(reason_logits)
            else:
                policy_logits, win_value, turns_value, reason_logits = self.model(obs_tensor)
                reason_probs = torch.sigmoid(reason_logits)

            # 在 GPU 端融合成单一连续张量 [count, 1861]，仅触发 1 次 D2H 显存拷贝与单个 NumPy 对象分配
            fused_tensor = torch.cat(
                [policy_logits, win_value, turns_value, reason_probs], dim=-1
            )
            return fused_tensor.float().cpu().numpy().reshape(-1)
