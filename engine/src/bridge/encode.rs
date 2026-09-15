use std::sync::LazyLock;

use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;
use crate::model::card::{CardAbility, CardColor, CardTier, JewelCard};
use crate::model::token::GemType;

/// 单张卡牌的特征维度 (33基础 + 5维ROI/效能 = 38)
pub const CARD_FEAT_DIM: usize = 38;

/// 单玩家手牌槽位数
pub const RESERVED_CARDS_SLOTS: usize = 3;

/// 单玩家仪表盘特征维度 (24基础资产与胜负进度 + 3槽手牌 * CARD_FEAT_DIM = 138)
pub const PLAYER_DASHBOARD_DIM: usize = 24 + RESERVED_CARDS_SLOTS * CARD_FEAT_DIM;

/// 观察向量维度 (9通道螺旋棋盘225 + 12市场卡(12*38=456) + 4王室 + 双方仪表盘(2*138=276) + 全局差值44 = 1005)
pub const OBS_SIZE: usize = 225 + 12 * CARD_FEAT_DIM + 4 + 2 * PLAYER_DASHBOARD_DIM + 44;

/// 动作空间大小（离散动作总维度）
pub const ACTION_SIZE: usize = 288;

/// 5x5 棋盘补盘顺时针螺旋排位归一化矩阵 (中心 (2,2) 为 0，向外顺时针扩展至 24)
pub const SPIRAL_RANK_MATRIX: [[f32; 5]; 5] = [
    [16.0 / 24.0, 15.0 / 24.0, 14.0 / 24.0, 13.0 / 24.0, 12.0 / 24.0],
    [17.0 / 24.0,  4.0 / 24.0,  3.0 / 24.0,  2.0 / 24.0, 11.0 / 24.0],
    [18.0 / 24.0,  5.0 / 24.0,  0.0 / 24.0,  1.0 / 24.0, 10.0 / 24.0],
    [19.0 / 24.0,  6.0 / 24.0,  7.0 / 24.0,  8.0 / 24.0,  9.0 / 24.0],
    [20.0 / 24.0, 21.0 / 24.0, 22.0 / 24.0, 23.0 / 24.0, 24.0 / 24.0],
];

/// 预计算 5x5 网格中所有 120 种可能的 2~3 连线几何线段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineDef {
    pub count: u8,
    pub positions: [(usize, usize); 3],
}

pub static ALL_LINES: LazyLock<Vec<LineDef>> = LazyLock::new(generate_all_line_definitions);

fn generate_all_line_definitions() -> Vec<LineDef> {
    let mut lines = Vec::with_capacity(120);
    const DIRS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

    for &(dr, dc) in DIRS.iter() {
        for r in 0..5isize {
            for c in 0..5isize {
                // 2 连线
                let r1 = r + dr;
                let c1 = c + dc;
                if (0..5).contains(&r1) && (0..5).contains(&c1) {
                    lines.push(LineDef {
                        count: 2,
                        positions: [
                            (r as usize, c as usize),
                            (r1 as usize, c1 as usize),
                            (0, 0),
                        ],
                    });

                    // 3 连线
                    let r2 = r1 + dr;
                    let c2 = c1 + dc;
                    if (0..5).contains(&r2) && (0..5).contains(&c2) {
                        lines.push(LineDef {
                            count: 3,
                            positions: [
                                (r as usize, c as usize),
                                (r1 as usize, c1 as usize),
                                (r2 as usize, c2 as usize),
                            ],
                        });
                    }
                }
            }
        }
    }
    lines
}

pub fn all_line_definitions() -> Vec<LineDef> {
    ALL_LINES.clone()
}

/// 状态特征张量编码器（当前玩家规范视角）
pub fn encode_state(state: &GameState) -> [f32; OBS_SIZE] {
    let mut out = [0.0f32; OBS_SIZE];
    let cp = state.current_player;
    let op = state.opponent_idx();

    // -------------------------------------------------------------
    // 分块 1: 5x5 棋盘空间 (25 格 × 9 通道 = 225 维) [0..225]
    // 包含 8 通道标记 One-Hot 与第 9 通道螺旋时空排位拓扑特征
    // -------------------------------------------------------------
    for r in 0..5 {
        for c in 0..5 {
            let base = (r * 5 + c) * 9;
            match state.board.get(r, c) {
                None => out[base] = 1.0,
                Some(GemType::White) => out[base + 1] = 1.0,
                Some(GemType::Blue) => out[base + 2] = 1.0,
                Some(GemType::Green) => out[base + 3] = 1.0,
                Some(GemType::Red) => out[base + 4] = 1.0,
                Some(GemType::Black) => out[base + 5] = 1.0,
                Some(GemType::Pearl) => out[base + 6] = 1.0,
                Some(GemType::Gold) => out[base + 7] = 1.0,
            }
            out[base + 8] = SPIRAL_RANK_MATRIX[r][c];
        }
    }

    // -------------------------------------------------------------
    // 分块 2: 金字塔市场卡牌 (12 槽位 × CARD_FEAT_DIM = 456 维) [225..681]
    // 剔除伪指示槽，严格保持真实可见卡牌实体，追加 5 维 ROI / 效能与动态净缺口
    // 槽位顺序与动作空间映射严格统一 (自底向上): Tier1 (5明), Tier2 (4明), Tier3 (3明)
    // -------------------------------------------------------------
    let mut offset = 225;
    // 12 个真实可见槽位: Tier1 (5明, 槽位 0..5), Tier2 (4明, 槽位 5..9), Tier3 (3明, 槽位 9..12)
    for &tier in &[CardTier::Tier1, CardTier::Tier2, CardTier::Tier3] {
        let t_idx = tier.index();
        let cap = tier.market_capacity();
        let cards = &state.pyramid[t_idx];

        for slot in 0..cap {
            let card_opt = cards.get(slot);
            encode_card_slot(
                &mut out[offset..offset + CARD_FEAT_DIM],
                card_opt,
                tier,
                &state.players[cp],
                Some(&state.players[op]),
            );
            offset += CARD_FEAT_DIM;
        }
    }

    // -------------------------------------------------------------
    // 分块 3: 场上王室卡 (4 维布尔向量) [681..685]
    // 指示 4 张固定属性王室卡是否已被拿取
    // -------------------------------------------------------------
    for royal_id in 0..4u8 {
        if state.royal_cards.iter().any(|rc| rc.id == royal_id) {
            out[offset + royal_id as usize] = 1.0;
        }
    }
    offset += 4;

    // -------------------------------------------------------------
    // 分块 4: 双方玩家仪表板 (2 玩家 × PLAYER_DASHBOARD_DIM = 276 维) [685..961]
    // -------------------------------------------------------------
    for (p_order, &p_idx) in [cp, op].iter().enumerate() {
        let p = &state.players[p_idx];
        let opp_of_p = if p_idx == cp { &state.players[op] } else { &state.players[cp] };
        let p_base = offset + p_order * PLAYER_DASHBOARD_DIM;

        // 标记库存 (8维，超限弃牌前可能短暂超过 10，钳位至 1.0)
        for i in 0..7 {
            out[p_base + i] = (p.tokens.counts[i] as f32 / 10.0).min(1.0);
        }
        out[p_base + 7] = (p.tokens.total() as f32 / 10.0).min(1.0);

        // 永久 Bonus (5维)
        for i in 0..5 {
            out[p_base + 8 + i] = (p.bonuses[i] as f32 / 6.0).min(1.0);
        }

        // 胜负条件进度 (8维，终局可能溢出，钳位至 1.0)
        out[p_base + 13] = (p.total_points as f32 / 20.0).min(1.0);
        out[p_base + 14] = (p.total_crowns as f32 / 10.0).min(1.0);
        let mut max_color = 0;
        for i in 0..5 {
            out[p_base + 15 + i] = (p.color_points[i] as f32 / 10.0).min(1.0);
            max_color = max_color.max(p.color_points[i]);
        }
        out[p_base + 20] = (max_color as f32 / 10.0).min(1.0);

        // 特权与王室指标 (3维)
        out[p_base + 21] = p.privileges as f32 / 3.0;
        out[p_base + 22] = p.royal_cards.len() as f32 / 2.0;
        out[p_base + 23] = if p.royals_claimed[0] { 1.0 } else { 0.0 };

        // 预留手牌 (3 槽位 × CARD_FEAT_DIM)
        for slot in 0..RESERVED_CARDS_SLOTS {
            let slot_start = p_base + 24 + slot * CARD_FEAT_DIM;
            let slot_slice = &mut out[slot_start..slot_start + CARD_FEAT_DIM];
            if let Some(rc) = p.reserved_cards.get(slot) {
                let is_self = p_idx == cp;
                if is_self || rc.is_public {
                    // 我方全部手牌，或对手公开明牌预留：写入完整 CARD_FEAT_DIM 维卡牌特征
                    encode_card_slot(
                        slot_slice,
                        Some(&rc.card),
                        rc.card.tier,
                        &state.players[p_idx],
                        Some(opp_of_p),
                    );
                } else {
                    // 对手盲抽暗牌 (M4): 保留 present 与 tier，其余私密属性掩蔽
                    encode_opponent_hidden_slot(slot_slice, rc.card.tier);
                }
            }
        }
    }
    offset += 2 * PLAYER_DASHBOARD_DIM;

    // -------------------------------------------------------------
    // 分块 5: 全局环境、博弈差值与胜负威胁 (44 维) [871..915]
    // -------------------------------------------------------------
    match state.phase {
        TurnPhase::OptionalActions => out[offset] = 1.0,
        TurnPhase::MandatoryAction => out[offset + 1] = 1.0,
        TurnPhase::CardAbilityJoker { .. } => out[offset + 2] = 1.0,
        TurnPhase::CardAbilitySameColor { .. } => out[offset + 3] = 1.0,
        TurnPhase::CardAbilitySteal => out[offset + 4] = 1.0,
        TurnPhase::SelectRoyalCard => out[offset + 5] = 1.0,
        TurnPhase::DiscardTokens => out[offset + 6] = 1.0,
        TurnPhase::SelectReserveGold => out[offset + 7] = 1.0,
        TurnPhase::GameOver(_) => out[offset + 8] = 1.0,
    }
    out[offset + 9] = state.privilege_pool as f32 / 3.0;
    out[offset + 10] = state.bag.len() as f32 / 25.0;

    // 11..17: 布袋中 7 种标记各自具体剩余数量 (7 维)
    let mut bag_counts = [0u8; 7];
    for &gem in &state.bag {
        bag_counts[gem.index()] += 1;
    }
    out[offset + 11] = bag_counts[0] as f32 / 4.0; // White
    out[offset + 12] = bag_counts[1] as f32 / 4.0; // Blue
    out[offset + 13] = bag_counts[2] as f32 / 4.0; // Green
    out[offset + 14] = bag_counts[3] as f32 / 4.0; // Red
    out[offset + 15] = bag_counts[4] as f32 / 4.0; // Black
    out[offset + 16] = bag_counts[5] as f32 / 2.0; // Pearl
    out[offset + 17] = bag_counts[6] as f32 / 3.0; // Gold

    // 18: 棋盘剩余标记数 / 25.0
    out[offset + 18] = state.board.count_tokens() as f32 / 25.0;
    // 19: 全局回合数归一化 turn_number / 80.0
    out[offset + 19] = (state.turn_number as f32 / 80.0).min(1.0);
    // 20: 额外回合标志位
    out[offset + 20] = if state.extra_turn_granted { 1.0 } else { 0.0 };
    // 21..23: 牌堆剩余比例
    out[offset + 21] = state.decks[0].len() as f32 / 30.0;
    out[offset + 22] = state.decks[1].len() as f32 / 24.0;
    out[offset + 23] = state.decks[2].len() as f32 / 13.0;

    let p_act = &state.players[cp];
    let p_opp = &state.players[op];

    // 24..27: 双方胜负指标差值归一化 (分差、皇冠差、单色差、特权差)
    out[offset + 24] = (p_act.total_points as f32 - p_opp.total_points as f32 + 20.0) / 40.0;
    out[offset + 25] = (p_act.total_crowns as f32 - p_opp.total_crowns as f32 + 10.0) / 20.0;
    let cp_max_c = p_act.color_points.iter().copied().max().unwrap_or(0);
    let op_max_c = p_opp.color_points.iter().copied().max().unwrap_or(0);
    out[offset + 26] = (cp_max_c as f32 - op_max_c as f32 + 10.0) / 20.0;
    out[offset + 27] = (p_act.privileges as f32 - p_opp.privileges as f32 + 3.0) / 6.0;

    // 28..29: 双方手牌余量 max(0, 10 - total) / 10.0
    out[offset + 28] = ((10.0 - p_act.tokens.total() as f32).max(0.0)) / 10.0;
    out[offset + 29] = ((10.0 - p_opp.tokens.total() as f32).max(0.0)) / 10.0;

    // 30..31: 双方手牌超限数量 max(0, total - 10) / 5.0
    out[offset + 30] = ((p_act.tokens.total() as f32 - 10.0).max(0.0)) / 5.0;
    out[offset + 31] = ((p_opp.tokens.total() as f32 - 10.0).max(0.0)) / 5.0;

    // 32..37: 双方胜利距离 (Gap to Win: 分数/20, 皇冠/10, 单色/10)
    let cp_pts_gap = (20.0 - p_act.total_points as f32).max(0.0) / 20.0;
    let op_pts_gap = (20.0 - p_opp.total_points as f32).max(0.0) / 20.0;
    let cp_crown_gap = (10.0 - p_act.total_crowns as f32).max(0.0) / 10.0;
    let op_crown_gap = (10.0 - p_opp.total_crowns as f32).max(0.0) / 10.0;
    let cp_col_gap = (10.0 - cp_max_c as f32).max(0.0) / 10.0;
    let op_col_gap = (10.0 - op_max_c as f32).max(0.0) / 10.0;

    out[offset + 32] = cp_pts_gap;
    out[offset + 33] = op_pts_gap;
    out[offset + 34] = cp_crown_gap;
    out[offset + 35] = op_crown_gap;
    out[offset + 36] = cp_col_gap;
    out[offset + 37] = op_col_gap;

    // 38..39: 双方离胜利的最小归一化差距
    let cp_min_gap = cp_pts_gap.min(cp_crown_gap).min(cp_col_gap);
    let op_min_gap = op_pts_gap.min(op_crown_gap).min(op_col_gap);
    out[offset + 38] = cp_min_gap;
    out[offset + 39] = op_min_gap;

    // 40: 我方当前是否存在即刻买卡斩杀动作 (Lethal)
    let cp_can_win_now = state
        .pyramid
        .iter()
        .flat_map(|row| row.iter())
        .chain(p_act.reserved_cards.iter().map(|rc| &rc.card))
        .any(|c| can_player_win_with_card(p_act, c));
    out[offset + 40] = if cp_can_win_now { 1.0 } else { 0.0 };

    // 41: 对手当前场上/公开手牌是否已经有买得起的致胜牌 (Opponent Lethal Threat)
    let op_can_win_now = state
        .pyramid
        .iter()
        .flat_map(|row| row.iter())
        .chain(
            p_opp
                .reserved_cards
                .iter()
                .filter(|rc| rc.is_public)
                .map(|rc| &rc.card),
        )
        .any(|c| can_player_win_with_card(p_opp, c));
    out[offset + 41] = if op_can_win_now { 1.0 } else { 0.0 };

    // 42: 本回合是否已补充棋盘 replenished_this_turn (1.0 或 0.0)
    out[offset + 42] = if state.replenished_this_turn { 1.0 } else { 0.0 };

    // 43: 本回合已消耗特权数 privileges_used_this_turn / 3.0
    out[offset + 43] = (state.privileges_used_this_turn as f32 / 3.0).min(1.0);

    out
}

/// 判定玩家是否能够通过购买某张卡牌直接达成 3 种胜负条件之一
#[inline]
fn can_player_win_with_card(p: &crate::game_state::player::PlayerState, c: &JewelCard) -> bool {
    if c.color == CardColor::Joker && !p.has_any_bonus() {
        return false;
    }
    if !p.can_afford(c) {
        return false;
    }
    if p.total_points + c.points >= 20 || p.total_crowns + c.crowns >= 10 {
        return true;
    }
    if let Some(gem) = c.color.to_gem_type() {
        if p.color_points[gem.index()] + c.points >= 10 {
            return true;
        }
    } else if c.color == CardColor::Joker && c.points > 0 {
        // Joker 变色卡可附着于已有 bonus 的任意颜色上
        for gem in GemType::BASIC_FIVE {
            if p.get_bonus(gem) > 0 && p.color_points[gem.index()] + c.points >= 10 {
                return true;
            }
        }
    }
    false
}

fn encode_card_slot(
    slice: &mut [f32],
    card_opt: Option<&JewelCard>,
    tier: CardTier,
    active_player: &crate::game_state::player::PlayerState,
    opp_player: Option<&crate::game_state::player::PlayerState>,
) {
    if let Some(c) = card_opt {
        slice[0] = 1.0; // present
        slice[1 + tier.index()] = 1.0; // tier
        slice[4] = c.points as f32 / 6.0;
        slice[5] = c.crowns as f32 / 3.0;

        // cost
        slice[6] = c.cost.white as f32 / 8.0;
        slice[7] = c.cost.blue as f32 / 8.0;
        slice[8] = c.cost.green as f32 / 8.0;
        slice[9] = c.cost.red as f32 / 8.0;
        slice[10] = c.cost.black as f32 / 8.0;
        slice[11] = c.cost.pearl as f32 / 2.0;

        // bonus_color
        match c.color {
            CardColor::White => slice[12] = 1.0,
            CardColor::Blue => slice[13] = 1.0,
            CardColor::Green => slice[14] = 1.0,
            CardColor::Red => slice[15] = 1.0,
            CardColor::Black => slice[16] = 1.0,
            CardColor::Points | CardColor::Joker => slice[17] = 1.0,
        }
        slice[18] = c.bonus as f32 / 2.0;

        // ability
        match c.ability {
            None => slice[19] = 1.0,
            Some(CardAbility::ExtraTurn) => slice[20] = 1.0,
            Some(CardAbility::TakePrivilege) => slice[21] = 1.0,
            Some(CardAbility::TakeSameColor) => slice[22] = 1.0,
            Some(CardAbility::StealToken) => slice[23] = 1.0,
            Some(CardAbility::ColorCopy) => slice[24] = 1.0,
            Some(CardAbility::ColorCopyAndExtraTurn) => slice[25] = 1.0,
        }

        slice[26] = if active_player.can_afford(c) {
            1.0
        } else {
            0.0
        };

        // 动态净缺口特征 (6 维: 5 基础宝石各自净缺口 + 珍珠缺口)
        // Deficit_gem = max(0, Cost_gem - Bonus_gem - Tokens_gem) / 8.0
        let basic_costs = [
            c.cost.white,
            c.cost.blue,
            c.cost.green,
            c.cost.red,
            c.cost.black,
        ];
        let mut sum_deficits = 0.0f32;
        for i in 0..5 {
            let cost_val = basic_costs[i] as f32;
            let bonus_val = active_player.bonuses[i] as f32;
            let token_val = active_player.tokens.counts[i] as f32;
            let deficit = (cost_val - bonus_val - token_val).max(0.0);
            slice[27 + i] = deficit / 8.0;
            sum_deficits += deficit;
        }
        // 珍珠缺口 (无 Bonus 折扣): max(0, Cost_pearl - Tokens_pearl) / 2.0
        let pearl_token = active_player.tokens.counts[GemType::Pearl.index()] as f32;
        let pearl_deficit = (c.cost.pearl as f32 - pearl_token).max(0.0);
        slice[32] = pearl_deficit / 2.0;
        sum_deficits += pearl_deficit;

        // -------------------------------------------------------------
        // 新增 5 维卡牌性价比与 ROI 效能特征 [33..38]
        // -------------------------------------------------------------
        let total_cost = (c.cost.white + c.cost.blue + c.cost.green + c.cost.red + c.cost.black + c.cost.pearl) as f32;
        // 33: 真实总费用强度 total_cost / 10.0
        slice[33] = (total_cost / 10.0).min(1.0);
        // 34: 声望投资回报率 points / max(1, total_cost) (范围 0.0..1.0)
        slice[34] = (c.points as f32 / total_cost.max(1.0)).min(1.0);
        // 35: 王冠投资回报率 crowns / max(1, total_cost) (范围 0.0..1.0)
        slice[35] = (c.crowns as f32 / total_cost.max(1.0)).min(1.0);

        // 36: 黄金冲抵后的实际有效缺口 max(0, sum_deficits - gold) / 8.0
        let gold_token = active_player.tokens.counts[GemType::Gold.index()] as f32;
        let effective_shortage = (sum_deficits - gold_token).max(0.0);
        slice[36] = (effective_shortage / 8.0).min(1.0);

        // 37: 对手当前是否买得起该卡 (防守/抢位战略价值)
        slice[37] = if let Some(opp) = opp_player {
            if opp.can_afford(c) { 1.0 } else { 0.0 }
        } else {
            0.0
        };
    }
}

/// 对手盲抽暗牌编码 (M4 POMDP 设计: 仅暴露 present 和公开可见的 tier，其余私密属性全部掩蔽为 0.0)
#[inline]
fn encode_opponent_hidden_slot(slice: &mut [f32], tier: CardTier) {
    slice[0] = 1.0; // present: 明确知晓对手此槽位持有暗牌
    slice[1 + tier.index()] = 1.0; // tier: 盲抽来源牌堆等级是公开操作 (Tier 1/2/3)
    // slice[4..CARD_FEAT_DIM] 保持 0.0 (费用、点数、皇冠、加成、技能、支付能力、缺口与 ROI 严格掩蔽)
}

/// 动作空间映射：将高层 Action 映射到 [0, 248] 离散 ID
pub fn action_to_id(action: &Action) -> usize {
    match action {
        Action::SkipOptional => 0,
        Action::UsePrivilege { r, c } => 1 + (r * 5 + c),
        Action::ReplenishBoard => 26,
        Action::TakeTokens { count, positions } => {
            if *count == 1 {
                let (r, c) = positions[0];
                27 + (r * 5 + c)
            } else {
                let idx = ALL_LINES
                    .iter()
                    .position(|l| {
                        l.count == *count
                            && l.positions[0] == positions[0]
                            && l.positions[1] == positions[1]
                            && (*count == 2 || l.positions[2] == positions[2])
                    })
                    .unwrap_or(0);
                52 + idx
            }
        }
        Action::ReserveCard { tier, slot } => match slot {
            Some(s) => {
                let tier_offset = match tier {
                    CardTier::Tier1 => 0,
                    CardTier::Tier2 => 5,
                    CardTier::Tier3 => 9,
                };
                172 + tier_offset + s
            }
            None => 184 + tier.index(),
        },
        Action::PurchaseCard {
            from_reserved,
            tier,
            slot,
        } => {
            if *from_reserved {
                199 + slot
            } else {
                let tier_offset = match tier {
                    CardTier::Tier1 => 0,
                    CardTier::Tier2 => 5,
                    CardTier::Tier3 => 9,
                };
                187 + tier_offset + slot
            }
        }
        Action::AssignJokerColor { color } => 202 + color.index(),
        Action::TakeSameColorToken { r, c } => 207 + (r * 5 + c),
        Action::StealToken { gem } => 232 + gem.index(),
        Action::SelectRoyal { royal_id } => 239 + (*royal_id as usize),
        Action::DiscardToken { gem } => 243 + gem.index(),
        Action::TakeGoldToken { r, c } => 250 + (r * 5 + c),
    }
}

/// 生成合法动作掩码 [bool; ACTION_SIZE]
pub fn action_mask(state: &GameState) -> [bool; ACTION_SIZE] {
    let legals = RuleEngine::legal_actions(state);
    action_mask_from_legals(&legals)
}

/// 基于已知合法动作列表生成动作掩码 (避免重复调用 RuleEngine::legal_actions)
#[inline]
pub fn action_mask_from_legals(legals: &[Action]) -> [bool; ACTION_SIZE] {
    let mut mask = [false; ACTION_SIZE];
    for act in legals {
        let id = action_to_id(act);
        if id < ACTION_SIZE {
            mask[id] = true;
        }
    }
    mask
}
