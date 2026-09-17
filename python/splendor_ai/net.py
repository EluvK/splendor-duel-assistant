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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]
        if mask is not None:
            # key padding mask: [B, 1, 1, N]
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = torch.where(key_mask, scores, -1e4)

        attn = F.softmax(scores, dim=-1)
        if mask is not None:
            # query mask: [B, 1, N, 1]，阻断无效 query 产生的 attention 扰动
            query_mask = mask.unsqueeze(1).unsqueeze(-1)
            attn = attn * query_mask.float()

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        out = self.norm(x + self.out_proj(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


def _generate_line_definitions() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成与 Rust encode.rs ALL_LINES 严格 1:1 一致的 120 条线段几何索引与属性张量."""
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    lines_pos = []     # [120, 3] 网格一维索引 (r*5 + c)
    lines_mask = []    # [120, 3] 有效位置掩码 (长度为 2 的线段第 3 位掩蔽为 0.0)
    lines_len = []     # [120] 0: 长度2, 1: 长度3
    lines_dir = []     # [120] 0: 横, 1: 竖, 2: 正斜, 3: 反斜

    for d_idx, (dr, dc) in enumerate(dirs):
        for r in range(5):
            for c in range(5):
                r1, c1 = r + dr, c + dc
                if 0 <= r1 < 5 and 0 <= c1 < 5:
                    # 2 连线
                    lines_pos.append([r * 5 + c, r1 * 5 + c1, 0])
                    lines_mask.append([1.0, 1.0, 0.0])
                    lines_len.append(0)
                    lines_dir.append(d_idx)

                    # 3 连线
                    r2, c2 = r1 + dr, c1 + dc
                    if 0 <= r2 < 5 and 0 <= c2 < 5:
                        lines_pos.append([r * 5 + c, r1 * 5 + c1, r2 * 5 + c2])
                        lines_mask.append([1.0, 1.0, 1.0])
                        lines_len.append(1)
                        lines_dir.append(d_idx)

    assert len(lines_pos) == 120
    mask_t = torch.tensor(lines_mask, dtype=torch.float32)
    weights = mask_t / mask_t.sum(dim=-1, keepdim=True)
    return (
        torch.tensor(lines_pos, dtype=torch.long),
        weights.view(1, 120, 3, 1),
        torch.tensor(lines_len, dtype=torch.long),
        torch.tensor(lines_dir, dtype=torch.long),
    )


def _generate_payment_plan_features() -> torch.Tensor:
    """生成与 Rust payment.rs PAYMENT_PLANS 严格一致的 84 种支付方案物理语义特征 [84, 7]."""
    plans = []
    # k = 0 (1 种): 无黄金替代
    plans.append([0, 0, 0, 0, 0, 0, 0])

    # k = 1 (6 种): 6 种颜色各选 1 枚替代
    for c0 in range(6):
        p = [0] * 6
        p[c0] += 1
        plans.append(p + [1])

    # k = 2 (21 种): 6 种颜色选 2 枚 (带放回)
    for c0 in range(6):
        for c1 in range(c0, 6):
            p = [0] * 6
            p[c0] += 1
            p[c1] += 1
            plans.append(p + [2])

    # k = 3 (56 种): 6 种颜色选 3 枚 (带放回)
    for c0 in range(6):
        for c1 in range(c0, 6):
            for c2 in range(c1, 6):
                p = [0] * 6
                p[c0] += 1
                p[c1] += 1
                p[c2] += 1
                plans.append(p + [3])

    assert len(plans) == 84
    arr = torch.tensor(plans, dtype=torch.float32)
    # 归一化: 6 色各自替代数 / 3.0, 消耗黄金总数 / 3.0
    arr[:, :6] = arr[:, :6] / 3.0
    arr[:, 6] = arr[:, 6] / 3.0
    return arr


class SplendorNet(nn.Module):
    """璀璨宝石：对决 策略-价值神经网络 (SplendorNet v3+).

    全新架构革新 (v3+ 评审强化版):
      1. 几何结构化与实体解耦跨模态策略头 (Geometry & Cross-Entity Policy Heads):
         - 空间几何连线 (172 维): 基于 5x5 网格卷积特征图与预计算 120 种线段局部 Pooling + 长度/方向嵌入打分
         - 黄金-卡牌低秩交互预留 (375 维): 25 黄金空间特征与 15 预留目标进行 q-k 双线性交互，并深度条件化全局态势 fused
         - 语义化方案购卡 (1260 维): 84 种支付方案通过物理结构嵌入编码，结合 15 槽位卡牌表征与 1/sqrt(d) 规范点积交互
         - 动作家族自适应标定 (Family Calibration): 4 大动作家族引入可学习尺度与偏置，平衡不同决策维度的 Logit 分布
      2. 严格空槽掩码与无偏集合自注意力 (Masked Set Attention & Category Embedding):
         - 引入 Empty-Slot Padding Mask 与 Masked Pooling，彻底杜绝未翻出卡牌或空手牌的虚假注意力模式
         - 采用 5 类语义类别嵌入 (Tier1/2/3 市场、我方手牌、敌方手牌)，完全保持同阶市场的置换等价性
      3. 固定支撑点分布价值头 (Categorical Distributional Value Head with Fixed Support):
         - 21 桶支撑点 [-1.0, 1.0]，使用软分类交叉熵训练，导出 ONNX 时闭式点积求期望，无缝兼容 Rust 推理
      4. 三重终局对称态势辅助头 (3D Victory Lead Head):
         - 严格对称覆盖终局声望差 (/25.0)、皇冠差 (/12.0) 与最大单色差 (/12.0)，为三重获胜线提供无遗漏的强方向梯度
      5. 固化多任务损失加权 (Fixed Multi-Task Loss Weighting):
         - 彻底消除自适应不确定性参数自动调大学习方差逃避拟合价值的问题
    """

    BOARD_CHANNELS = 9                         # 8 标记 + 1 螺旋 Rank
    BOARD_GRID = 5
    CARD_FEAT_DIM = 36                         # 30 基础特征 + 6 维缺口与博弈效能
    NUM_CARD_ENTITIES = 18                     # 12 市场明牌 + 3 我方手牌 + 3 敌方手牌
    NUM_CARD_CATEGORIES = 5                    # 5 类: Tier1, Tier2, Tier3, MyReserved, OppReserved
    RESERVED_CARDS_SLOTS = 3
    PLAYER_BASE_DIM = 24
    PLAYER_DASHBOARD_DIM = PLAYER_BASE_DIM + RESERVED_CARDS_SLOTS * CARD_FEAT_DIM  # 24 + 3 * 36 = 132
    GLOBAL_CTX_DIM = 44                        # 38 基础环境与差值 + 6 维 pending_resource
    OBS_SIZE = 225 + 12 * CARD_FEAT_DIM + 4 + 2 * PLAYER_DASHBOARD_DIM + GLOBAL_CTX_DIM  # 969
    ACTION_SIZE = SplendorDuelEnv.ACTION_SIZE  # 1856

    STATE_LEADS_SLICE = slice(947, 950)        # 声望分差 [947]、皇冠差 [948]、单色差 [949] 在观测向量中的标准切片
    WIN_CLASSES = 2                            # 胜负二分类输出 [P(win), P(loss)]
    REASON_CLASSES = 6                         # 胜因二分类输出 [我方20分, 10冠, 10单色; 敌方20分, 10冠, 10单色]

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

        # 2. 共享卡牌实体编码器 + 类别等价偏置 + 集合自注意力
        self.card_encoder = nn.Sequential(
            nn.Linear(self.CARD_FEAT_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, card_embed_dim),
            nn.LayerNorm(card_embed_dim),
            nn.ReLU(),
        )
        self.card_category_emb = nn.Embedding(self.NUM_CARD_CATEGORIES, card_embed_dim)
        # 18 个槽位的语义类别: Tier1 (5), Tier2 (4), Tier3 (3), 我方预留 (3), 敌方预留 (3)
        category_indices = torch.tensor(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            dtype=torch.long,
        )
        self.register_buffer("category_indices", category_indices, persistent=False)

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

        # 5. 几何结构化与实体解耦跨模态策略打分器 (Cross-Entity Policy Heads)

        # --- (A) [0..171] 172维: 特权、拿标记、几何连线动作 ---
        # 棋盘单格特征融合 (16 局部卷积 + 256 全局 fused -> 32，无动态 Expand 纯原生广播)
        self.grid_conv_proj = nn.Linear(16, 32)
        self.grid_fused_proj = nn.Linear(fusion_hidden, 32)

        self.take_single_scorer = nn.Linear(32, 1)        # 25 个单格宝石打分
        self.privilege_scorer = nn.Linear(32, 1)          # 25 个特权拿宝石打分
        self.skip_scorer = nn.Linear(fusion_hidden, 1)    # 1 维跳过可选动作
        self.replenish_scorer = nn.Linear(fusion_hidden, 1) # 1 维补盘动作

        # 120 种几何连线汇聚打分
        lines_pos, lines_weights, lines_len, lines_dir = _generate_line_definitions()
        self.register_buffer("lines_pos", lines_pos, persistent=False)
        self.register_buffer("lines_weights", lines_weights, persistent=False)
        self.register_buffer("lines_len", lines_len, persistent=False)
        self.register_buffer("lines_dir", lines_dir, persistent=False)

        self.line_cell_scorer = nn.Linear(32, 1)
        self.line_len_scorer = nn.Embedding(2, 1)
        self.line_dir_scorer = nn.Embedding(4, 1)

        # --- (B) [172..546] 375维: 预留卡牌动作头 (25 黄金坐标 x 15 目标) ---
        self.reserve_gold_scorer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.reserve_gold_q = nn.Linear(32, 16, bias=False)

        self.reserve_market_scorer_c = nn.Linear(card_embed_dim, 1)
        self.reserve_market_scorer_f = nn.Linear(fusion_hidden, 1)
        self.reserve_market_k_c = nn.Linear(card_embed_dim, 16)
        self.reserve_market_k_f = nn.Linear(fusion_hidden, 16)

        self.reserve_blind_scorer = nn.Sequential(
            nn.Linear(fusion_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.reserve_blind_k = nn.Sequential(
            nn.Linear(fusion_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 16),
        )
        self.reserve_global_bias = nn.Linear(fusion_hidden, 1)

        # --- (C) [547..1806] 1260维: 购买卡牌动作头 (15 槽位 x 84 支付方案) ---
        plan_feats = _generate_payment_plan_features()
        self.register_buffer("payment_plan_features", plan_feats, persistent=False)
        self.plan_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.buy_card_scorer_c = nn.Linear(card_embed_dim, 1)
        self.buy_card_scorer_f = nn.Linear(fusion_hidden, 1)
        self.buy_plan_scorer = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 84),
        )
        self.buy_card_factor_c = nn.Linear(card_embed_dim, 32, bias=False)
        self.buy_card_factor_f = nn.Linear(fusion_hidden, 32, bias=False)

        # --- (D) [1807..1855] 49维: Joker、同色、偷标记、王室、弃牌等后续能力动作头 ---
        self.ability_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 49),
        )

        # --- 动作家族尺度与基线校准 (Family Calibration: tokens, reserve, buy, ability) ---
        self.family_scales = nn.Parameter(torch.ones(4))
        self.family_biases = nn.Parameter(torch.zeros(4))

        # 6. 多任务评估头
        # Win Head: 2 维分类 Logits [P(win), P(loss)] (正统 AlphaZero 设计，天然概率语义且杜绝 Tanh 饱和)
        self.win_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.WIN_CLASSES),
        )

        # Turns Head: [0.0, 1.0]
        self.turns_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Reason Head: 6 分类区分归属胜因 Logits [我方20分, 10冠, 10单色; 敌方20分, 10冠, 10单色]
        self.reason_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.REASON_CLASSES),
        )

        # 7. 三重胜负线态势差辅助头 (Score Lead Head: 声望差 / 25.0, 皇冠差 / 12.0, 单色差 / 12.0)
        self.lead_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3),
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

        # 3. 卡牌实体提取、掩码保护与集合自注意力 (共 18 实体: 12 市场 + 3 我方预留 + 3 敌方预留)
        market_cards = market_cards_flat.view(b_size, 12, self.CARD_FEAT_DIM)
        reserved_cards = reserved_cards_flat.view(b_size, 3, self.CARD_FEAT_DIM)
        opp_reserved_cards = opp_reserved_flat.view(b_size, 3, self.CARD_FEAT_DIM)
        all_cards = torch.cat([market_cards, reserved_cards, opp_reserved_cards], dim=1)  # [B, 18, 36]

        # 构造有效卡牌掩码 (第 0 通道 present > 0.5)
        valid_mask = all_cards[:, :, 0] > 0.5  # [B, 18]

        # 卡牌类别嵌入加成 (保持同一 Tier 内部的置换等价性)
        card_embeddings = self.card_encoder(all_cards) + self.card_category_emb(self.category_indices)  # [B, 18, 128]
        card_tokens = self.set_attention(card_embeddings, mask=valid_mask)  # [B, 18, 128]

        # Masked Pooling (严格消除空槽的无效特征干扰)
        mask_exp = valid_mask.unsqueeze(-1)  # [B, 18, 1]
        masked_tokens = torch.where(mask_exp, card_tokens, -1e4)
        max_pool = masked_tokens.max(dim=1)[0]
        max_pool = torch.where(valid_mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))

        tokens_for_sum = torch.where(mask_exp, card_tokens, torch.zeros_like(card_tokens))
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        mean_pool = tokens_for_sum.sum(dim=1) / valid_counts
        x_card = self.card_pool_fc(torch.cat([mean_pool, max_pool], dim=-1))  # [B, 128]

        # 4. 上下文标量特征
        context = torch.cat([royals, self_base, opp_base, global_ctx], dim=-1)  # [B, 92]
        x_context = self.context_mlp(context)  # [B, 128]

        # 5. 全局融合
        fused = self.fusion(torch.cat([x_board, x_card, x_context], dim=-1))  # [B, 256]

        # 6. 结构化解耦动作打分 (Cross-Entity Policy Heads)

        # --- (A) [0..171] 172维: 特权、拿标记、空间几何连线动作 ---
        board_grid_feat = board_conv_16.view(b_size, 16, 25).permute(0, 2, 1)  # [B, 25, 16]
        cell_features = F.relu(self.grid_conv_proj(board_grid_feat) + self.grid_fused_proj(fused).unsqueeze(1))  # [B, 25, 32]

        skip_score = self.skip_scorer(fused)                                    # [B, 1]
        privilege_scores = self.privilege_scorer(cell_features).squeeze(-1)     # [B, 25]
        replenish_score = self.replenish_scorer(fused)                          # [B, 1]
        single_scores = self.take_single_scorer(cell_features).squeeze(-1)       # [B, 25]

        # 几何连线汇聚 (利用预计算的 lines_weights 进行纯静态矩阵运算，无动态 Expand 节点)
        line_cells = cell_features[:, self.lines_pos, :]                        # [B, 120, 3, 32]
        line_pooled = (line_cells * self.lines_weights).sum(dim=2)              # [B, 120, 32]
        line_scores = (
            self.line_cell_scorer(line_pooled)
            + self.line_len_scorer(self.lines_len).unsqueeze(0)
            + self.line_dir_scorer(self.lines_dir).unsqueeze(0)
        ).squeeze(-1)  # [B, 120]

        token_logits = torch.cat(
            [skip_score, privilege_scores, replenish_score, single_scores, line_scores],
            dim=-1,
        )  # [B, 172]

        # --- (B) [172..546] 375维: 预留卡牌动作 (25 黄金网格 x 15 目标: 12 市场明牌 + 3 盲抽) ---
        gold_scores = self.reserve_gold_scorer(cell_features)                   # [B, 25, 1]
        gold_q = self.reserve_gold_q(cell_features)                             # [B, 25, 16]

        market_cards_12 = card_tokens[:, :12, :]                                # [B, 12, 128]
        market_target_s = self.reserve_market_scorer_c(market_cards_12) + self.reserve_market_scorer_f(fused).unsqueeze(1)  # [B, 12, 1]
        market_target_k = self.reserve_market_k_c(market_cards_12) + self.reserve_market_k_f(fused).unsqueeze(1)          # [B, 12, 16]

        blind_target_s = self.reserve_blind_scorer(fused).view(b_size, 3, 1)    # [B, 3, 1]
        blind_target_k = self.reserve_blind_k(fused).view(b_size, 3, 16)        # [B, 3, 16]

        target_scores = torch.cat([market_target_s, blind_target_s], dim=1).transpose(1, 2)  # [B, 1, 15]
        target_k = torch.cat([market_target_k, blind_target_k], dim=1)          # [B, 15, 16]

        unary_reserve = gold_scores + target_scores                             # [B, 25, 15]
        interaction_reserve = torch.matmul(gold_q, target_k.transpose(-2, -1)) * 0.25  # [B, 25, 15] (缩放 1/sqrt(16))
        global_r_bias = self.reserve_global_bias(fused).unsqueeze(1)           # [B, 1, 1]

        reserve_grid = unary_reserve + interaction_reserve + global_r_bias      # [B, 25, 15]
        reserve_logits = reserve_grid.contiguous().view(b_size, 375)

        # --- (C) [547..1806] 1260维: 购买卡牌动作 (15 槽位 x 84 支付方案) ---
        active_15_cards = card_tokens[:, :15, :]                                # [B, 15, 128] (12 市场明牌 + 3 我方手牌)
        card_buy_s = self.buy_card_scorer_c(active_15_cards) + self.buy_card_scorer_f(fused).unsqueeze(1)  # [B, 15, 1]
        plan_s = self.buy_plan_scorer(fused).unsqueeze(1)                       # [B, 1, 84]

        card_buy_f = self.buy_card_factor_c(active_15_cards) + self.buy_card_factor_f(fused).unsqueeze(1)  # [B, 15, 32]
        plan_emb = self.plan_encoder(self.payment_plan_features)                # [84, 32]

        buy_interaction = torch.matmul(card_buy_f, plan_emb.t()) / 5.656854    # [B, 15, 84] (缩放 1/sqrt(32))
        buy_grid = card_buy_s + plan_s + buy_interaction                        # [B, 15, 84]
        buy_logits = buy_grid.contiguous().view(b_size, 1260)

        # --- (D) [1807..1855] 49维: 后续能力动作 ---
        ability_logits = self.ability_head(fused)                               # [B, 49]

        # --- 动作家族自适应标定 (Family Scale & Bias Calibration) ---
        token_logits = token_logits * self.family_scales[0] + self.family_biases[0]
        reserve_logits = reserve_logits * self.family_scales[1] + self.family_biases[1]
        buy_logits = buy_logits * self.family_scales[2] + self.family_biases[2]
        ability_logits = ability_logits * self.family_scales[3] + self.family_biases[3]

        assembled_logits = torch.cat(
            [token_logits, reserve_logits, buy_logits, ability_logits],
            dim=-1,
        )  # [B, 1856]

        # 7. 多任务估值
        win_logits = self.win_head(fused)  # [B, 2]
        win_probs = F.softmax(win_logits, dim=-1)  # [B, 2]: [P(win), P(loss)]
        win_value = win_probs[:, 0:1] - win_probs[:, 1:2]  # [B, 1] 严格落在 [-1.0, 1.0]

        turns_value = self.turns_head(fused)  # [B, 1]
        reason_logits = self.reason_head(fused)  # [B, 6]
        lead_value = self.lead_head(fused)  # [B, 3]

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
        """从状态张量中精确提取双方声望差、皇冠差与最大单色差 (分差/25.0, 皇冠差/12.0, 单色差/12.0) 作为辅助任务目标."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs[:, cls.STATE_LEADS_SLICE].clamp(-1.0, 1.0)

    @staticmethod
    def compute_win_target_distribution(target_win: torch.Tensor) -> torch.Tensor:
        """将连续或离散胜率目标标量 ([-1.0, 1.0]) 转换为 [P(win), P(loss)] 二分类概率目标分布."""
        if target_win.dim() == 1:
            target_win = target_win.unsqueeze(-1)
        p_win = (target_win.clamp(-1.0, 1.0) + 1.0) * 0.5
        return torch.cat([p_win, 1.0 - p_win], dim=-1)

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
