"""Policy-Value Neural Network for Splendor Duel."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from splendor_ai.env import SplendorDuelEnv


class ResidualBlock2D(nn.Module):
    """2D 空间残差卷积块."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class SplendorNet(nn.Module):
    """璀璨宝石：对决 Policy-Value 神经网络.

    输入: 726 维扁平状态观察向量
      - 前 200 维拆解为 (B, 8, 5, 5) 棋盘空间网格，经由 2D ResNet 提取 3 连相邻几何特征
      - 后 526 维经由多层感知机 (MLP) 提取卡牌市场、双方手牌与胜负节奏特征
    输出:
      - policy_logits: [B, 288] 动作概率对数
      - value: [B, 1] 行动方胜率预测 ([-1.0, 1.0])
    """

    OBS_SIZE = SplendorDuelEnv.OBS_SIZE       # 726
    ACTION_SIZE = SplendorDuelEnv.ACTION_SIZE # 288
    BOARD_CHANNELS = 8
    BOARD_GRID = 5

    def __init__(
        self,
        spatial_channels: int = 64,
        num_res_blocks: int = 2,
        context_hidden: int = 256,
        fusion_hidden: int = 256,
    ) -> None:
        super().__init__()

        # 1. 棋盘 2D 卷积骨干
        self.board_conv_in = nn.Sequential(
            nn.Conv2d(self.BOARD_CHANNELS, spatial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(),
        )
        self.board_res_blocks = nn.ModuleList(
            [ResidualBlock2D(spatial_channels) for _ in range(num_res_blocks)]
        )
        self.board_pool = nn.AvgPool2d(kernel_size=3, stride=2)
        board_out_dim = spatial_channels * 2 * 2  # 64 * 4 = 256

        # 2. 上下文标量特征 MLP 骨干
        context_in_dim = self.OBS_SIZE - 200  # 525
        self.context_mlp = nn.Sequential(
            nn.Linear(context_in_dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
        )

        # 3. 融合主干 (Fusion Trunk)
        self.fusion = nn.Sequential(
            nn.Linear(board_out_dim + context_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
        )

        # 4. 双头输出
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, self.ACTION_SIZE),
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            obs: [B, 725] 或 [725] 状态张量.

        Returns:
            policy_logits: [B, 256] 未掩码的动作 logits
            value: [B, 1] 胜率预测 ([-1.0, 1.0])
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 拆分特征
        board_flat = obs[:, :200]
        context = obs[:, 200:]

        # 重构 5x5x8 为 (B, 8, 5, 5)
        b_size = obs.shape[0]
        board = board_flat.view(b_size, self.BOARD_GRID, self.BOARD_GRID, self.BOARD_CHANNELS)
        board = board.permute(0, 3, 1, 2).contiguous()

        # 空间前向
        x_board = self.board_conv_in(board)
        for block in self.board_res_blocks:
            x_board = block(x_board)
        x_board = self.board_pool(x_board).view(b_size, -1)

        # 上下文前向
        x_context = self.context_mlp(context)

        # 特征融合
        fused = self.fusion(torch.cat([x_board, x_context], dim=-1))

        # 双头输出
        logits = self.policy_head(fused)
        value = self.value_head(fused)

        return logits, value

    @staticmethod
    def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用合法动作掩码，将非法动作 logits 赋为极小值 (-1e4，适配 AMP float16 范围)."""
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return torch.where(mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))

    def predict_action_probs(
        self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算合法动作概率分布与胜率评估.

        Args:
            obs: [B, 725] 状态
            mask: [B, 256] 动作掩码 (bool)
            temperature: 采样温度 (默认 1.0)

        Returns:
            probs: [B, 256] 合法动作概率分布
            value: [B, 1] 胜率估计
        """
        logits, value = self.forward(obs)
        if mask is not None:
            logits = self.mask_logits(logits, mask)

        if temperature <= 1e-4:
            # 贪婪模式 (argmax)
            probs = F.one_hot(logits.argmax(dim=-1), num_classes=self.ACTION_SIZE).float()
        else:
            probs = F.softmax(logits / temperature, dim=-1)

        return probs, value

    def export_onnx_bytes(self) -> bytes:
        """将当前模型导出为 ONNX 二进制字节流 (供 Rust tract-onnx 引擎极速推理)."""
        import io

        orig_device = next(self.parameters()).device
        dummy_obs = torch.zeros(1, self.OBS_SIZE, dtype=torch.float32, device=orig_device)
        buf = io.BytesIO()

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                torch.onnx.export(
                    self,
                    dummy_obs,
                    buf,
                    input_names=["obs"],
                    output_names=["policy_logits", "value"],
                    dynamic_axes={
                        "obs": {0: "batch_size"},
                        "policy_logits": {0: "batch_size"},
                        "value": {0: "batch_size"},
                    },
                    opset_version=17,
                    do_constant_folding=True,
                )
        finally:
            if was_training:
                self.train()

        return buf.getvalue()
