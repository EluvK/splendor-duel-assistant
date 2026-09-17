"""Policy-Value Neural Network (SplendorNet v3) for Splendor Duel."""

from typing import Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from splendor_ai.env import SplendorDuelEnv


class ResidualBlock2D(nn.Module):
    """2D 空间残差卷积块 (基于 LayerNorm / GroupNorm 特征保持，无小 Batch BN 抖动)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
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
    """璀璨宝石：对决 策略-价值神经网络 (SplendorNet v3).

    全新架构革新 (v3):
      1. 实体解耦跨模态策略头 (Cross-Entity Policy Head):
         - 购卡打分 (1260 维): 解耦为 15 槽位卡牌 Token 独立打分 + 84 种支付方案经济偏好 + 低秩双线性交互
         - 预留打分 (375 维): 解耦为 25 黄金网格空间特征 + 15 预留目标特征结合
         - 彻底消除全局 256 维向量单层硬猜 1260 维的严重信息瓶颈与槽位遗忘
      2. 分位数胜率价值头 (Two-Hot Distributional Win Head):
         - 21 桶支撑点 [-1.0, 1.0]，使用软分类交叉熵训练，杜绝 Tanh 饱和区梯度消失与 MSE 均方误差震荡
         - 导出 ONNX 时内部自动点积计算数学期望标量 [-1.0, 1.0]，无缝兼容 Rust tract 引擎
      3. 博弈态势差辅助头 (Score & Crown Lead Head):
         - 辅助重构声望差 (/25.0) 与皇冠差 (/12.0)，强化中盘模糊局面下主干表征的胜负态势感知
      4. 固化多任务损失加权 (Fixed Multi-Task Loss Weighting):
         - 彻底消除自适应不确定性参数自动调大学习方差逃避拟合价值的问题
    """

    BOARD_CHANNELS = 9                         # 8 标记 + 1 螺旋 Rank
    BOARD_GRID = 5
    CARD_FEAT_DIM = 36                         # 30 基础特征 + 6 维缺口与博弈效能
    NUM_CARD_ENTITIES = 18                     # 12 市场明牌 + 3 我方手牌 + 3 敌方手牌
    RESERVED_CARDS_SLOTS = 3
    PLAYER_BASE_DIM = 24
    PLAYER_DASHBOARD_DIM = PLAYER_BASE_DIM + RESERVED_CARDS_SLOTS * CARD_FEAT_DIM  # 24 + 3 * 36 = 132
    GLOBAL_CTX_DIM = 44                        # 38 基础环境与差值 + 6 维 pending_resource
    OBS_SIZE = 225 + 12 * CARD_FEAT_DIM + 4 + 2 * PLAYER_DASHBOARD_DIM + GLOBAL_CTX_DIM  # 969
    ACTION_SIZE = SplendorDuelEnv.ACTION_SIZE  # 1856

    STATE_LEADS_SLICE = slice(947, 949)        # 声望分差 [947] 与皇冠差 [948] 在观测向量中的标准切片
    NUM_SUPPORT_BINS = 21                      # Two-Hot 离散分桶数

    def __init__(
        self,
        spatial_channels: int = 64,
        num_res_blocks: int = 2,
        card_embed_dim: int = 128,
        context_hidden: int = 128,
        fusion_hidden: int = 256,
    ) -> None:
        super().__init__()

        # 1. 棋盘 2D 卷积骨干 (保持 5x5 全分辨率，采用 GroupNorm 消除批次大小波动)
        self.board_conv_in = nn.Sequential(
            nn.Conv2d(self.BOARD_CHANNELS, spatial_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, spatial_channels),
            nn.ReLU(),
        )
        self.board_res_blocks = nn.ModuleList(
            [ResidualBlock2D(spatial_channels) for _ in range(num_res_blocks)]
        )
        self.board_conv_out = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=1, bias=False),
            nn.GroupNorm(4, 16),
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
        context_in_dim = 4 + self.PLAYER_BASE_DIM + self.PLAYER_BASE_DIM + self.GLOBAL_CTX_DIM
        self.context_mlp = nn.Sequential(
            nn.Linear(context_in_dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
        )

        # 4. 特征融合主干 (棋盘 128 + 全局卡牌 128 + 上下文 128 = 384 维)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + context_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
        )

        # 5. 实体解耦跨模态策略打分器 (Cross-Entity Policy Heads)
        # [0..171] 172维: 特权、拿标记、连线动作
        self.token_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 172),
        )

        # [172..546] 375维: 预留卡牌动作头 (25 黄金网格坐标 x 15 目标)
        # 黄金网格空间打分: 16通道局部网格 -> 1 维打分
        self.reserve_gold_scorer = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # 12 市场明牌目标打分
        self.reserve_market_scorer = nn.Sequential(
            nn.Linear(card_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # 3 盲抽牌堆顶目标打分 (直接由全局态势打分，无动态 Expand 算子)
        self.reserve_blind_scorer = nn.Sequential(
            nn.Linear(fusion_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.reserve_global_bias = nn.Linear(fusion_hidden, 1)

        # [547..1806] 1260维: 购买卡牌动作头 (15 槽位 x 84 支付方案)
        # 1. 槽位卡牌购买意愿打分 (15 槽位共享)
        self.buy_card_scorer = nn.Sequential(
            nn.Linear(card_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # 2. 全局支付方案经济偏好打分 (84 方案)
        self.buy_plan_scorer = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 84),
        )
        # 3. 卡牌-方案低秩双线性交互
        self.buy_card_factor = nn.Linear(card_embed_dim, 32, bias=False)
        self.plan_embeddings = nn.Parameter(torch.randn(32, 84) * 0.02)

        # [1807..1855] 49维: Joker、同色、偷标记、王室、弃牌等后续能力动作头
        self.ability_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 49),
        )

        # 6. 多任务评估头
        # Win Head: 21 桶 Two-Hot 分位数分布
        self.win_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.NUM_SUPPORT_BINS),
        )
        self.register_buffer(
            "support_points",
            torch.linspace(-1.0, 1.0, self.NUM_SUPPORT_BINS),
            persistent=False,
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

        # 7. 博弈态势差辅助头 (Score & Crown Lead Head: 声望分差 / 25.0, 皇冠差 / 12.0)
        self.lead_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def _forward_impl(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        board_flat = obs[:, :idx_board_end]
        market_cards_flat = obs[:, idx_board_end:idx_market_end]
        royals = obs[:, idx_market_end:idx_royals_end]
        self_base = obs[:, idx_royals_end:idx_self_base_end]
        reserved_cards_flat = obs[:, idx_self_base_end:idx_self_res_end]
        opp_base = obs[:, idx_self_res_end:idx_opp_base_end]
        opp_reserved_flat = obs[:, idx_opp_base_end:idx_opp_res_end]
        global_ctx = obs[:, idx_opp_res_end:idx_global_end]

        # 2. 棋盘空间前向
        board = board_flat.view(b_size, self.BOARD_GRID, self.BOARD_GRID, self.BOARD_CHANNELS)
        board = board.permute(0, 3, 1, 2).contiguous()
        x_board_conv = self.board_conv_in(board)
        for block in self.board_res_blocks:
            x_board_conv = block(x_board_conv)
        board_conv_16 = self.board_conv_out(x_board_conv)  # [B, 16, 5, 5]
        x_board = self.board_fc(board_conv_16.view(b_size, -1))  # [B, 128]

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

        # 4. 上下文标量特征
        context = torch.cat([royals, self_base, opp_base, global_ctx], dim=-1)  # [B, 92]
        x_context = self.context_mlp(context)  # [B, 128]

        # 5. 全局融合
        fused = self.fusion(torch.cat([x_board, x_card, x_context], dim=-1))  # [B, 256]

        # 6. 结构化解耦动作打分
        # (A) [0..171] 172维: 特权、拿标记、连线动作
        token_logits = self.token_head(fused)

        # (B) [172..546] 375维: 预留卡牌动作 (25 黄金网格 x 15 目标: 12 市场明牌 + 3 盲抽)
        board_grid_feat = board_conv_16.view(b_size, 16, 25).permute(0, 2, 1)  # [B, 25, 16]
        gold_scores = self.reserve_gold_scorer(board_grid_feat)  # [B, 25, 1]

        market_targets_s = self.reserve_market_scorer(card_tokens[:, :12, :]).squeeze(-1)  # [B, 12]
        blind_targets_s = self.reserve_blind_scorer(fused)  # [B, 3]
        target_scores = torch.cat([market_targets_s, blind_targets_s], dim=-1).unsqueeze(1)  # [B, 1, 15]

        global_r_bias = self.reserve_global_bias(fused).unsqueeze(1)  # [B, 1, 1]
        reserve_grid = gold_scores + target_scores + global_r_bias  # [B, 25, 15]
        reserve_logits = reserve_grid.contiguous().view(b_size, 375)

        # (C) [547..1806] 1260维: 购买卡牌动作 (15 槽位 x 84 支付方案)
        active_15_cards = card_tokens[:, :15, :]  # [B, 15, 128] (12 市场明牌 + 3 我方手牌)
        card_buy_s = self.buy_card_scorer(active_15_cards)  # [B, 15, 1]
        plan_s = self.buy_plan_scorer(fused).unsqueeze(1)  # [B, 1, 84]

        card_buy_f = self.buy_card_factor(active_15_cards)  # [B, 15, 32]
        buy_interaction = torch.matmul(card_buy_f, self.plan_embeddings)  # [B, 15, 84]

        buy_grid = card_buy_s + plan_s + buy_interaction  # [B, 15, 84]
        buy_logits = buy_grid.contiguous().view(b_size, 1260)

        # (D) [1807..1855] 49维: 后续能力动作
        ability_logits = self.ability_head(fused)

        # 拼装回完整的 1856 维动作空间
        assembled_logits = torch.cat(
            [token_logits, reserve_logits, buy_logits, ability_logits],
            dim=-1,
        )

        # 7. 多任务估值
        win_logits = self.win_head(fused)  # [B, 21]
        win_probs = F.softmax(win_logits, dim=-1)
        support_col = self.support_points.view(-1, 1).to(obs.device)
        win_value = torch.matmul(win_probs, support_col)  # [B, 1]

        turns_value = self.turns_head(fused)  # [B, 1]
        reason_logits = self.reason_head(fused)  # [B, 3]
        lead_value = self.lead_head(fused)  # [B, 2]

        return assembled_logits, win_value, turns_value, reason_logits, win_logits, lead_value

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 (默认返回标准 4 元组，与 ONNX 导出和推理签名 100% 对齐)."""
        assembled_logits, win_value, turns_value, reason_logits, _, _ = self._forward_impl(obs)
        return assembled_logits, win_value, turns_value, reason_logits

    def forward_train(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练专用前向传播，返回包含未聚合 win_logits 与 lead_pred 的 6 元组."""
        return self._forward_impl(obs)

    def compute_multi_task_loss(
        self,
        policy_loss: torch.Tensor,
        win_loss: torch.Tensor,
        turns_loss: torch.Tensor,
        reason_loss: torch.Tensor,
        lead_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """固化多任务损失加权 (保证价值头满额梯度，杜绝方差崩塌)."""
        w_p = 1.0
        w_w = 1.0
        w_t = 0.30
        w_r = 0.15
        w_l = 0.20
        total_loss = w_p * policy_loss + w_w * win_loss + w_t * turns_loss + w_r * reason_loss
        if lead_loss is not None:
            total_loss = total_loss + w_l * lead_loss
        weights = {
            "w_policy": w_p,
            "w_win": w_w,
            "w_turns": w_t,
            "w_reason": w_r,
            "w_lead": w_l,
        }
        return total_loss, weights

    def compute_homoscedastic_loss(
        self,
        policy_loss: torch.Tensor,
        win_loss: torch.Tensor,
        turns_loss: torch.Tensor,
        reason_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """向后兼容接口，映射至固化多任务损失."""
        return self.compute_multi_task_loss(policy_loss, win_loss, turns_loss, reason_loss)

    @classmethod
    def extract_state_leads(cls, obs: torch.Tensor) -> torch.Tensor:
        """从状态张量中精确提取双方声望差与皇冠差 (分差/25.0, 皇冠差/12.0) 作为辅助任务目标."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs[:, cls.STATE_LEADS_SLICE].clamp(-1.0, 1.0)

    @staticmethod
    def to_two_hot(target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """将连续胜率目标标量 ([-1.0, 1.0]) 转换为 21 桶 Two-Hot 软分类目标分布."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)

        low = support[0].item()
        high = support[-1].item()
        num_bins = support.shape[0]
        clamped = target.clamp(low, high)

        span = high - low
        float_idx = ((clamped - low) / span) * (num_bins - 1)
        float_idx = float_idx.clamp(0.0, float(num_bins - 1))

        lower_idx = float_idx.floor().long().clamp(0, num_bins - 2)
        upper_idx = lower_idx + 1
        weight_upper = float_idx - lower_idx.float()
        weight_lower = 1.0 - weight_upper

        b_size = target.shape[0]
        dist = torch.zeros(b_size, num_bins, device=target.device, dtype=target.dtype)
        dist.scatter_add_(1, lower_idx, weight_lower)
        dist.scatter_add_(1, upper_idx, weight_upper)
        return dist

    @staticmethod
    def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用合法动作掩码，将非法动作 logits 赋为极小值 (-1e4，适配 AMP float16 范围)."""
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return torch.where(mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))

    def predict_action_probs(
        self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算合法动作概率分布与多目标评估 (保持 API 统一)."""
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
