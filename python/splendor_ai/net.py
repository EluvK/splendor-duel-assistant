"""Policy-Value Neural Network (SplendorNet v2) for Splendor Duel."""

from typing import Optional, Tuple, Dict
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


class SetSelfAttention(nn.Module):
    """用于卡牌实体交互的紧凑无偏集合自注意力机制 (纯矩阵乘法，完全兼容 ONNX / tract 极速编译)."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] (N=15, D=128)
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        out = self.norm(x + self.out_proj(out))
        return out


class SplendorNet(nn.Module):
    """璀璨宝石：对决 Policy-Value 神经网络 (SplendorNet v2).

    高级架构特性:
      1. 棋盘拓扑感知 (9 通道): 引入螺旋 Rank 归一化拓扑通道，全分辨率 ResNet2D 提取空间相邻与连线特征
      2. 共享卡牌编码器 (CardEncoder): 15 个卡牌实体 (12 市场明牌 + 3 我方手牌，33 维含动态净缺口) 共享参数
      3. 集合自注意力机制 (Set Attention): 捕捉卡牌间互相提供加成/购买先后的连带协同
      4. 结构化动作打分器 (Structured Policy Head):
         - 卡牌预留/购买动作通过全局表征与对应卡牌 Token 双线性点积生成
         - 离散与连线动作经由专用感知头生成，按 288 维标准动作空间装配
      5. 解耦多任务输出与同方差不确定性自适应损失 (Homoscedastic Loss Weighting)
    """

    BOARD_CHANNELS = 9                         # 8 标记 + 1 螺旋 Rank
    BOARD_GRID = 5
    CARD_FEAT_DIM = 36                         # 30 基础特征 + 6 维缺口与博弈效能
    NUM_CARD_ENTITIES = 18                     # 12 市场明牌 + 3 我方手牌 + 3 敌方手牌
    RESERVED_CARDS_SLOTS = 3
    PLAYER_BASE_DIM = 24
    PLAYER_DASHBOARD_DIM = PLAYER_BASE_DIM + RESERVED_CARDS_SLOTS * CARD_FEAT_DIM  # 24 + 3 * 36 = 132
    GLOBAL_CTX_DIM = 62                        # 40 基础环境与差值 + 22 维 Pending Decision Context
    OBS_SIZE = 225 + 12 * CARD_FEAT_DIM + 4 + 2 * PLAYER_DASHBOARD_DIM + GLOBAL_CTX_DIM  # 987
    ACTION_SIZE = SplendorDuelEnv.ACTION_SIZE  # 288

    def __init__(
        self,
        spatial_channels: int = 64,
        num_res_blocks: int = 2,
        card_embed_dim: int = 128,
        context_hidden: int = 128,
        fusion_hidden: int = 256,
    ) -> None:
        super().__init__()

        # 1. 棋盘 2D 卷积骨干 (保持 5x5 全分辨率)
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
            nn.Linear(16 * self.BOARD_GRID * self.BOARD_GRID, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 2. 共享卡牌实体编码器 + 槽位类型偏置 + 集合自注意力
        self.card_encoder = nn.Sequential(
            nn.Linear(self.CARD_FEAT_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, card_embed_dim),
            nn.LayerNorm(card_embed_dim),
            nn.ReLU(),
        )
        self.slot_type_emb = nn.Embedding(self.NUM_CARD_ENTITIES, card_embed_dim)
        self.set_attention = SetSelfAttention(embed_dim=card_embed_dim, num_heads=4)

        # 全局卡牌池化网络
        self.card_pool_fc = nn.Sequential(
            nn.Linear(card_embed_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 3. 标量上下文 MLP 骨干
        # 上下文包含: 王室卡(4) + 我方基础(24) + 敌方基础(24) + 全局差值环境(GLOBAL_CTX_DIM=40) = 92 维
        context_in_dim = 4 + self.PLAYER_BASE_DIM + self.PLAYER_BASE_DIM + self.GLOBAL_CTX_DIM
        self.context_mlp = nn.Sequential(
            nn.Linear(context_in_dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
        )

        # 4. 特征融合主干
        # 棋盘(128) + 全局卡牌(128) + 上下文(128) = 384 维
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + context_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
        )

        # 5. 结构化动作打分器 (Structured Policy Head)
        # (A) 卡牌实体打分投影: 通过双线性点积与卡牌 Token 交互
        self.reserve_card_proj = nn.Linear(fusion_hidden, card_embed_dim)
        self.buy_market_proj = nn.Linear(fusion_hidden, card_embed_dim)
        self.buy_reserved_proj = nn.Linear(fusion_hidden, card_embed_dim)
        # 卡牌策略打分缩放因子 1 / sqrt(d) 与可学习增益参数，平衡双线性点积与 MLP 离散头的 Logits 尺度
        self.card_logit_scale = card_embed_dim ** -0.5
        self.card_logit_gain = nn.Parameter(torch.ones(1))

        # (B) 其余离散与连线动作打分头 (共 261 维):
        # 0..171 (172维: 特权、拿标记、连线), 184..186 (3维: 盲抽), 202..287 (86维: Joker、弃牌、拿黄金等)
        self.num_discrete_actions = self.ACTION_SIZE - 27  # 288 - 27 = 261
        self.discrete_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, self.num_discrete_actions),
        )

        # 6. 多任务评估头
        # Win Head: [-1.0, 1.0]
        self.win_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        # Turns Head: [0.0, 1.0]
        self.turns_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        # Reason Head: 3 分类独立胜因 Logits [20_pts, 10_crowns, 10_color]
        self.reason_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        # 7. 同方差不确定性多任务可学习参数 log(sigma^2): [policy, win, turns, reason]
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            obs: [B, 1005] 或 [1005] 状态张量.

        Returns:
            policy_logits: [B, 288] 组装好的未掩码动作 logits
            win_value: [B, 1] 纯胜率期望 ([-1.0, 1.0])
            turns_value: [B, 1] 归一化剩余轮数预期 ([0.0, 1.0])
            reason_logits: [B, 3] 终局胜因 3 维独立多标签 logits
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        b_size = obs.shape[0]

        # 1. 动态自适应切分特征分块
        c = self.CARD_FEAT_DIM
        p_base = self.PLAYER_BASE_DIM
        p_res = self.RESERVED_CARDS_SLOTS * c

        idx_board_end = 225
        idx_market_end = idx_board_end + 12 * c
        idx_royals_end = idx_market_end + 4
        idx_self_base_end = idx_royals_end + p_base
        idx_self_res_end = idx_self_base_end + p_res
        idx_opp_base_end = idx_self_res_end + p_base
        idx_opp_res_end = idx_opp_base_end + p_res
        idx_global_end = idx_opp_res_end + self.GLOBAL_CTX_DIM

        # [0..225]: 5x5 网格 x 9 通道
        board_flat = obs[:, :idx_board_end]
        # [225..idx_market_end]: 市场 12 张卡 x CARD_FEAT_DIM 维
        market_cards_flat = obs[:, idx_board_end:idx_market_end]
        # [idx_market_end..idx_royals_end]: 场上王室卡 (4维)
        royals = obs[:, idx_market_end:idx_royals_end]
        # [idx_royals_end..idx_self_base_end]: 我方基础特征 (24维)
        self_base = obs[:, idx_royals_end:idx_self_base_end]
        # [idx_self_base_end..idx_self_res_end]: 我方预留卡 3 张 x CARD_FEAT_DIM 维
        reserved_cards_flat = obs[:, idx_self_base_end:idx_self_res_end]
        # [idx_self_res_end..idx_opp_base_end]: 敌方基础特征 (24维)
        opp_base = obs[:, idx_self_res_end:idx_opp_base_end]
        # [idx_opp_base_end..idx_opp_res_end]: 敌方预留卡 3 张 x CARD_FEAT_DIM 维
        opp_reserved_flat = obs[:, idx_opp_base_end:idx_opp_res_end]
        # [idx_opp_res_end:]: 全局环境与差值 (40维)
        global_ctx = obs[:, idx_opp_res_end:idx_global_end]

        # 2. 棋盘空间前向
        board = board_flat.view(b_size, self.BOARD_GRID, self.BOARD_GRID, self.BOARD_CHANNELS)
        board = board.permute(0, 3, 1, 2).contiguous()
        x_board = self.board_conv_in(board)
        for block in self.board_res_blocks:
            x_board = block(x_board)
        x_board = self.board_conv_out(x_board).view(b_size, -1)
        x_board = self.board_fc(x_board)  # [B, 128]

        # 3. 卡牌实体池提取与自注意力 (共 18 实体: 12 市场 + 3 我方预留 + 3 敌方预留)
        market_cards = market_cards_flat.view(b_size, 12, self.CARD_FEAT_DIM)
        reserved_cards = reserved_cards_flat.view(b_size, 3, self.CARD_FEAT_DIM)
        opp_reserved_cards = opp_reserved_flat.view(b_size, 3, self.CARD_FEAT_DIM)
        all_cards = torch.cat([market_cards, reserved_cards, opp_reserved_cards], dim=1)  # [B, 18, 36]

        slot_indices = torch.arange(self.NUM_CARD_ENTITIES, device=obs.device).unsqueeze(0)
        card_embeddings = self.card_encoder(all_cards) + self.slot_type_emb(slot_indices)  # [B, 18, 128]
        card_tokens = self.set_attention(card_embeddings)  # [B, 18, 128]

        # 全局卡牌聚合
        mean_pool = card_tokens.mean(dim=1)
        max_pool = card_tokens.max(dim=1)[0]
        x_card = self.card_pool_fc(torch.cat([mean_pool, max_pool], dim=-1))  # [B, 128]

        # 4. 上下文标量特征 (王室卡4 + 我方基础24 + 敌方基础24 + 全局环境40 = 92维)
        context = torch.cat([royals, self_base, opp_base, global_ctx], dim=-1)  # [B, 92]
        x_context = self.context_mlp(context)  # [B, 128]

        # 5. 全局融合
        fused = self.fusion(torch.cat([x_board, x_card, x_context], dim=-1))  # [B, 256]

        # 6. 结构化动作打分
        card_scale = self.card_logit_scale * self.card_logit_gain

        # (A) 卡牌相关操作 (应用缩放 1 / sqrt(d) 与可学习增益)
        # 预留市场卡 12 张: [172..183]
        q_reserve = self.reserve_card_proj(fused)  # [B, 128]
        logits_reserve = torch.einsum("bd,bnd->bn", q_reserve, card_tokens[:, :12]) * card_scale  # [B, 12]

        # 购买市场卡 12 张: [187..198]
        q_buy_market = self.buy_market_proj(fused)  # [B, 128]
        logits_buy_market = torch.einsum("bd,bnd->bn", q_buy_market, card_tokens[:, :12]) * card_scale  # [B, 12]

        # 购买预留卡 3 张: [199..201]
        q_buy_reserved = self.buy_reserved_proj(fused)  # [B, 128]
        logits_buy_reserved = torch.einsum("bd,bnd->bn", q_buy_reserved, card_tokens[:, 12:15]) * card_scale  # [B, 3]

        # (B) 其余离散动作 (261 维)
        discrete_logits = self.discrete_head(fused)  # [B, 261]

        # (C) 拼装回完整的 288 维动作空间 (严格保持向后兼容性)
        assembled_logits = torch.cat(
            [
                discrete_logits[:, :172],       # [0..171]: Skip, Privilege, Replenish, TakeTokens (172维)
                logits_reserve,                 # [172..183]: Reserve Market (12维)
                discrete_logits[:, 172:175],    # [184..186]: Reserve Blind Tier 1/2/3 (3维)
                logits_buy_market,              # [187..198]: Purchase Market (12维)
                logits_buy_reserved,            # [199..201]: Purchase Reserved (3维)
                discrete_logits[:, 175:],       # [202..287]: Joker, Steal, Royal, Discard, Gold (86维)
            ],
            dim=-1,
        )

        # 7. 多任务估值
        win_value = self.win_head(fused)
        turns_value = self.turns_head(fused)
        reason_logits = self.reason_head(fused)

        return assembled_logits, win_value, turns_value, reason_logits

    def compute_homoscedastic_loss(
        self,
        policy_loss: torch.Tensor,
        win_loss: torch.Tensor,
        turns_loss: torch.Tensor,
        reason_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """同方差任务不确定性自适应多任务损失加权 (平滑恒正正则项).

        Loss = 0.5 * exp(-s_p) * L_p + 0.5 * exp(-s_w) * L_w + 0.5 * exp(-s_t) * L_t
             + 0.5 * exp(-s_r) * L_r + 0.5 * sum(log(1 + exp(s_i)))
        采用 log(1 + sigma^2) 保证正则项恒为正，杜绝方差崩溃与负 Loss 异常，完美适配健康度监控.
        """
        s_p, s_w, s_t, s_r = self.log_vars[0], self.log_vars[1], self.log_vars[2], self.log_vars[3]

        # 软下界恒正正则项: 0.5 * sum(log(1 + exp(s)))
        reg = 0.5 * (
            torch.log1p(torch.exp(s_p))
            + torch.log1p(torch.exp(s_w))
            + torch.log1p(torch.exp(s_t))
            + torch.log1p(torch.exp(s_r))
        )

        total_loss = (
            0.5 * torch.exp(-s_p) * policy_loss
            + 0.5 * torch.exp(-s_w) * win_loss
            + 0.5 * torch.exp(-s_t) * turns_loss
            + 0.5 * torch.exp(-s_r) * reason_loss
            + reg
        )

        weights = {
            "w_policy": (0.5 * torch.exp(-s_p)).item(),
            "w_win": (0.5 * torch.exp(-s_w)).item(),
            "w_turns": (0.5 * torch.exp(-s_t)).item(),
            "w_reason": (0.5 * torch.exp(-s_r)).item(),
        }
        return total_loss, weights

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
            obs: [B, 915] 状态
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
