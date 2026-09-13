use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;
use crate::model::card::{CardAbility, CardColor, CardTier, JewelCard, RoyalAbility};
use crate::model::token::GemType;

/// 观察向量维度
pub const OBS_SIZE: usize = 725;

/// 动作空间大小（离散动作总维度）
pub const ACTION_SIZE: usize = 256;

/// 预计算 5x5 网格中所有 120 种可能的 2~3 连线几何线段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineDef {
    pub count: u8,
    pub positions: [(usize, usize); 3],
}

pub fn all_line_definitions() -> Vec<LineDef> {
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

/// 状态特征张量编码器（当前玩家规范视角）
pub fn encode_state(state: &GameState) -> [f32; OBS_SIZE] {
    let mut out = [0.0f32; OBS_SIZE];
    let cp = state.current_player;
    let op = state.opponent_idx();

    // -------------------------------------------------------------
    // 分块 1: 5x5 棋盘空间 (25 格 × 8 通道 = 200 维) [0..200]
    // -------------------------------------------------------------
    for r in 0..5 {
        for c in 0..5 {
            let base = (r * 5 + c) * 8;
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
        }
    }

    // -------------------------------------------------------------
    // 分块 2: 金字塔市场卡牌 (15 槽位 × 27 维 = 405 维) [200..605]
    // -------------------------------------------------------------
    let mut offset = 200;
    // 15 个槽位: Tier3 (3明+1余), Tier2 (4明+1余), Tier1 (5明+1余)
    for &tier in &[CardTier::Tier3, CardTier::Tier2, CardTier::Tier1] {
        let t_idx = tier.index();
        let cap = tier.market_capacity();
        let cards = &state.pyramid[t_idx];

        for slot in 0..cap {
            let card_opt = cards.get(slot);
            encode_card_slot(&mut out[offset..offset + 27], card_opt, tier, &state.players[cp]);
            offset += 27;
        }

        // 牌堆指示槽（slot == cap）
        let has_deck = !state.decks[t_idx].is_empty();
        if has_deck {
            out[offset] = 1.0; // present
            out[offset + 1 + t_idx] = 1.0; // tier
            // 牌堆剩余比例
            let max_deck = match tier {
                CardTier::Tier1 => 30.0,
                CardTier::Tier2 => 24.0,
                CardTier::Tier3 => 13.0,
            };
            out[offset + 4] = state.decks[t_idx].len() as f32 / max_deck;
        }
        offset += 27;
    }

    // -------------------------------------------------------------
    // 分块 3: 场上王室卡 (4 槽位 × 5 维 = 20 维) [605..625]
    // -------------------------------------------------------------
    for royal_id in 0..4u8 {
        let base = offset + (royal_id as usize) * 5;
        if let Some(r) = state.royal_cards.iter().find(|rc| rc.id == royal_id) {
            out[base] = 1.0; // available
            out[base + 1] = r.points as f32 / 3.0;
            match r.ability {
                Some(RoyalAbility::StealToken) => out[base + 2] = 1.0,
                Some(RoyalAbility::TakePrivilege) => out[base + 3] = 1.0,
                Some(RoyalAbility::ExtraTurn) => out[base + 4] = 1.0,
                None => {}
            }
        }
    }
    offset += 20;

    // -------------------------------------------------------------
    // 分块 4: 双方玩家状态 (2 玩家 × 42 维 = 84 维) [625..709]
    // -------------------------------------------------------------
    for (p_order, &p_idx) in [cp, op].iter().enumerate() {
        let p = &state.players[p_idx];
        let p_base = offset + p_order * 42;

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

        // 预留手牌 (3 槽位 × 6 维 = 18维)
        for slot in 0..3 {
            let slot_base = p_base + 24 + slot * 6;
            if let Some(card) = p.reserved_cards.get(slot) {
                out[slot_base] = 1.0;
                out[slot_base + 1] = 1.0; // public
                out[slot_base + 2] = card.points as f32 / 6.0;
                out[slot_base + 3] = card.crowns as f32 / 3.0;
                if let Some(gem) = card.color.to_gem_type() {
                    out[slot_base + 4] = gem.index() as f32 / 5.0;
                }
                out[slot_base + 5] = if state.players[cp].can_afford(card) {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
    offset += 84;

    // -------------------------------------------------------------
    // 分块 5: 全局环境与阶段 (16 维) [709..725]
    // -------------------------------------------------------------
    match state.phase {
        TurnPhase::OptionalActions => out[offset] = 1.0,
        TurnPhase::MandatoryAction => out[offset + 1] = 1.0,
        TurnPhase::CardAbilityJoker { .. } => out[offset + 2] = 1.0,
        TurnPhase::CardAbilitySameColor { .. } => out[offset + 3] = 1.0,
        TurnPhase::CardAbilitySteal => out[offset + 4] = 1.0,
        TurnPhase::SelectRoyalCard => out[offset + 5] = 1.0,
        TurnPhase::DiscardTokens => out[offset + 6] = 1.0,
        TurnPhase::GameOver(_) => out[offset + 7] = 1.0,
    }
    out[offset + 8] = state.privilege_pool as f32 / 3.0;
    out[offset + 9] = state.bag.len() as f32 / 25.0;
    out[offset + 10] = state.board.count_tokens() as f32 / 25.0;
    out[offset + 11] = (state.turn_number as f32 / 60.0).min(1.0);
    out[offset + 12] = if state.extra_turn_granted { 1.0 } else { 0.0 };
    out[offset + 13] = state.decks[0].len() as f32 / 30.0;
    out[offset + 14] = state.decks[1].len() as f32 / 24.0;
    out[offset + 15] = state.decks[2].len() as f32 / 13.0;

    out
}

fn encode_card_slot(
    slice: &mut [f32],
    card_opt: Option<&JewelCard>,
    tier: CardTier,
    active_player: &crate::game_state::player::PlayerState,
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
    }
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
                let all_lines = all_line_definitions();
                let idx = all_lines
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
    }
}

/// 生成合法动作掩码 [bool; 256]
pub fn action_mask(state: &GameState) -> [bool; ACTION_SIZE] {
    let mut mask = [false; ACTION_SIZE];
    let legals = RuleEngine::legal_actions(state);
    for act in legals {
        let id = action_to_id(&act);
        if id < ACTION_SIZE {
            mask[id] = true;
        }
    }
    mask
}
