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

    输入: OBS_SIZE (879) 维扁平状态观察向量
      - 前 200 维拆解为 (B, 8, 5, 5) 棋盘空间网格，经由 2D ResNet + 1x1 Conv 提取全分辨率 3 连相邻几何特征
      - 后 679 维经由多层感知机 (MLP) 提取卡牌市场、双方手牌、胜负紧迫度与相对博弈差值特征
    输出:
      - policy_logits: [B, 288] 动作概率对数
      - win_value: [B, 1] 纯胜率预期 ([-1.0, 1.0])
      - turns_value: [B, 1] 归一化剩余轮数预期 ([0.0, 1.0])
      - reason_logits: [B, 3] 终局多标签独立胜因 Logits ([20_pts, 10_crowns, 10_color])
    """

    OBS_SIZE = SplendorDuelEnv.OBS_SIZE       # 879
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

        # 1. 棋盘 2D 卷积骨干 (保持 5x5 全分辨率，避免不对称池化模糊连线拓扑)
        self.board_conv_in = nn.Sequential(
            nn.Conv2d(self.BOARD_CHANNELS, spatial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(),
        )
        self.board_res_blocks = nn.ModuleList(
            [ResidualBlock2D(spatial_channels) for _ in range(num_res_blocks)]
        )
        self.board_conv_out = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.board_fc = nn.Sequential(
            nn.Linear(16 * self.BOARD_GRID * self.BOARD_GRID, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        board_out_dim = 256

        # 2. 上下文标量特征 MLP 骨干
        context_in_dim = self.OBS_SIZE - 200  # 542
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

        # 4. 多任务输出头 (解耦策略、胜负、剩余回合与终局胜因)
        # (1) Policy Head: 288 维合法动作 Logits
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, self.ACTION_SIZE),
        )

        # (2) Win Head: 纯胜率预期 [-1.0, 1.0] (不带时间折现，消除负折扣苟活漏洞)
        self.win_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        # (3) Turns Head: 归一化剩余轮数预期 [0.0, 1.0] (0..80 轮，推动 MCTS 压榨步数速胜)
        self.turns_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # (4) Reason Head: 终局胜因 3 分类独立多标签 Logits [20_pts, 10_crowns, 10_color]
        self.reason_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            obs: [B, 742] 或 [742] 状态张量.

        Returns:
            policy_logits: [B, 288] 未掩码动作 logits
            win_value: [B, 1] 纯胜率期望 ([-1.0, 1.0])
            turns_value: [B, 1] 归一化剩余轮数预期 ([0.0, 1.0])
            reason_logits: [B, 3] 终局胜因 3 维独立多标签 logits
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

        # 空间前向 (1x1 Conv + Flatten + FC 全分辨率拓扑保持)
        x_board = self.board_conv_in(board)
        for block in self.board_res_blocks:
            x_board = block(x_board)
        x_board = self.board_conv_out(x_board).view(b_size, -1)
        x_board = self.board_fc(x_board)

        # 上下文前向
        x_context = self.context_mlp(context)

        # 特征融合
        fused = self.fusion(torch.cat([x_board, x_context], dim=-1))

        # 多头输出
        policy_logits = self.policy_head(fused)
        win_value = self.win_head(fused)
        turns_value = self.turns_head(fused)
        reason_logits = self.reason_head(fused)

        return policy_logits, win_value, turns_value, reason_logits

    @staticmethod
    def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用合法动作掩码，将非法动作 logits 赋为极小值 (-1e4，适配 AMP float16 范围)."""
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return torch.where(mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))

    def predict_action_probs(
        self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算合法动作概率分布与多目标评估.

        Args:
            obs: [B, 742] 状态
            mask: [B, 288] 动作掩码 (bool)
            temperature: 采样温度 (默认 1.0)

        Returns:
            probs: [B, 288] 合法动作概率分布
            win_value: [B, 1] 纯胜率预期 ([-1.0, 1.0])
            turns_value: [B, 1] 归一化剩余轮数预期 ([0.0, 1.0])
            reason_logits: [B, 3] 终局胜因 Logits
        """
        policy_logits, win_value, turns_value, reason_logits = self.forward(obs)
        if mask is not None:
            policy_logits = self.mask_logits(policy_logits, mask)

        if temperature <= 1e-4:
            probs = F.one_hot(policy_logits.argmax(dim=-1), num_classes=self.ACTION_SIZE).float()
        else:
            probs = F.softmax(policy_logits / temperature, dim=-1)

        return probs, win_value, turns_value, reason_logits

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
                    output_names=["policy_logits", "win_value", "turns_value", "reason_logits"],
                    dynamic_axes={
                        "obs": {0: "batch_size"},
                        "policy_logits": {0: "batch_size"},
                        "win_value": {0: "batch_size"},
                        "turns_value": {0: "batch_size"},
                        "reason_logits": {0: "batch_size"},
                    },
                    opset_version=17,
                    do_constant_folding=True,
                )
        finally:
            if was_training:
                self.train()

        return buf.getvalue()
