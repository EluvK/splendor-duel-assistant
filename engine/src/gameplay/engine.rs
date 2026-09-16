use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use super::payment::{check_payment_divergence, compute_card_payment, compute_custom_payment};
use super::scoring::check_victory;
use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::model::action::Action;
use crate::model::card::{CardAbility, CardColor, CardTier, JewelCard, RoyalAbility};
use crate::model::token::{GemType, TokenCollection};

/// 游戏执行引擎
pub struct GameEngine;

impl GameEngine {
    /// 执行动作，推进状态机
    pub fn step(state: &mut GameState, action: &Action) -> Result<(), String> {
        match action {
            Action::SkipOptional => Self::step_skip_optional(state),
            Action::UsePrivilege { r, c } => Self::step_use_privilege(state, *r, *c),
            Action::ReplenishBoard => Self::step_replenish_board(state),
            Action::TakeTokens { count, positions } => {
                Self::step_take_tokens(state, *count, *positions)
            }
            Action::ReserveCard { tier, slot } => Self::step_reserve_card(state, *tier, *slot),
            Action::TakeGoldToken { r, c } => Self::step_take_gold_token(state, *r, *c),
            Action::PurchaseCard {
                from_reserved,
                tier,
                slot,
            } => Self::step_purchase_card(state, *from_reserved, *tier, *slot),
            Action::ConfirmPayment => Self::step_confirm_payment(state),
            Action::PayGoldFor { gem } => Self::step_pay_gold_for(state, *gem),
            Action::AssignJokerColor { color } => Self::step_assign_joker(state, *color),
            Action::TakeSameColorToken { r, c } => Self::step_take_same_color(state, *r, *c),
            Action::StealToken { gem } => Self::step_steal_token(state, *gem),
            Action::SelectRoyal { royal_id } => Self::step_select_royal(state, *royal_id),
            Action::DiscardToken { gem } => Self::step_discard_token(state, *gem),
        }
    }

    fn step_skip_optional(state: &mut GameState) -> Result<(), String> {
        if state.phase == TurnPhase::OptionalActions {
            state.phase = TurnPhase::MandatoryAction;
            Ok(())
        } else if state.phase == TurnPhase::MandatoryAction {
            Self::after_action_check(state);
            Ok(())
        } else {
            Err("Not in OptionalActions or MandatoryAction phase".into())
        }
    }

    fn step_use_privilege(state: &mut GameState, r: usize, c: usize) -> Result<(), String> {
        if state.phase != TurnPhase::OptionalActions {
            return Err("Not in OptionalActions phase".into());
        }
        if state.replenished_this_turn {
            return Err("Cannot use privilege after replenishing board this turn".into());
        }
        let current_player = state.current_player;
        if state.players[current_player].privileges == 0 {
            return Err("No privilege scrolls available to use".into());
        }

        let gem = state.board.get(r, c).ok_or("No token at position")?;
        if gem.is_gold() {
            return Err("Cannot take gold with privilege".into());
        }

        // 归还特权卷轴到公用池
        state.return_privilege_from(current_player);

        // 取走标记
        state.board.take(r, c);
        state.players[current_player].tokens.add(gem, 1);
        state.privileges_used_this_turn = state.privileges_used_this_turn.saturating_add(1);

        Ok(())
    }

    fn step_replenish_board(state: &mut GameState) -> Result<(), String> {
        if state.phase != TurnPhase::OptionalActions && state.phase != TurnPhase::MandatoryAction {
            return Err("Cannot replenish board in current phase".into());
        }
        if state.replenished_this_turn {
            return Err("Cannot replenish board more than once per turn".into());
        }
        if state.bag.is_empty() || !state.board.has_empty_slot() {
            return Err("Cannot replenish board: bag empty or board full".into());
        }

        // 摇匀布袋后沿螺旋填入棋盘 (基于对局生命周期衍生独立高熵种子)
        let step_seed = state.next_rng_seed();
        let mut rng = ChaCha8Rng::seed_from_u64(step_seed);
        state.bag.shuffle(&mut rng);
        state.board.fill_spiral(&mut state.bag);

        // 对手获得 1 特权卷轴
        let opponent = state.opponent_idx();
        state.grant_privilege_to(opponent);
        state.replenished_this_turn = true;

        // 若处于 OptionalActions，补充棋盘后可选行动全部结束（特权不可再用），直接进入 MandatoryAction 阶段
        // 若处于 MandatoryAction（因无合法行动被迫补板），补板后留在 MandatoryAction
        if state.phase == TurnPhase::OptionalActions {
            state.phase = TurnPhase::MandatoryAction;
        }
        Ok(())
    }

    fn step_take_tokens(
        state: &mut GameState,
        count: u8,
        positions: [(usize, usize); 3],
    ) -> Result<(), String> {
        if state.phase != TurnPhase::MandatoryAction {
            return Err("Not in MandatoryAction phase".into());
        }
        if count == 0 || count > 3 {
            return Err("Invalid token count".into());
        }

        let mut taken = Vec::with_capacity(3);
        for i in 0..count as usize {
            let (r, c) = positions[i];
            let gem = state
                .board
                .take(r, c)
                .ok_or_else(|| format!("No token at ({r}, {c})"))?;
            if gem.is_gold() {
                return Err("Cannot take gold via TakeTokens".into());
            }
            taken.push(gem);
        }

        let current_player = state.current_player;
        for &gem in taken.iter() {
            state.players[current_player].tokens.add(gem, 1);
        }

        // 惩罚特权检查：3 同色 或 2 珍珠
        let should_grant_privilege = if count == 3 && taken[0] == taken[1] && taken[1] == taken[2] {
            true
        } else {
            let pearls = taken.iter().filter(|&&g| g == GemType::Pearl).count();
            pearls >= 2
        };

        if should_grant_privilege {
            let opponent = state.opponent_idx();
            state.grant_privilege_to(opponent);
        }

        Self::after_action_check(state);
        Ok(())
    }

    fn step_take_gold_token(state: &mut GameState, r: usize, c: usize) -> Result<(), String> {
        if state.phase != TurnPhase::MandatoryAction {
            return Err("Not in MandatoryAction phase".into());
        }
        let current_player = state.current_player;
        if state.players[current_player].reserved_cards.len() >= 3 {
            return Err("Reserve limit reached (max 3)".into());
        }

        let gem = state.board.get(r, c).ok_or("No token at position")?;
        if !gem.is_gold() {
            return Err("Selected token is not gold".into());
        }

        state.board.take(r, c);
        state.players[current_player].tokens.add(GemType::Gold, 1);

        // 拿取黄金后，进入选择预留卡牌阶段（金字塔市场牌尚未动，牌堆顶也未翻开）
        state.phase = TurnPhase::SelectReserveCard;
        Ok(())
    }

    fn step_reserve_card(
        state: &mut GameState,
        tier: CardTier,
        slot: Option<usize>,
    ) -> Result<(), String> {
        if state.phase != TurnPhase::SelectReserveCard {
            return Err("Not in SelectReserveCard phase".into());
        }
        let current_player = state.current_player;
        if state.players[current_player].reserved_cards.len() >= 3 {
            return Err("Reserve limit reached (max 3)".into());
        }

        // 预留卡牌
        let (card, is_public) = match slot {
            Some(s) => {
                let tier_idx = tier.index();
                if s >= state.pyramid[tier_idx].len() {
                    return Err("Invalid pyramid slot".into());
                }
                let c = state.pyramid[tier_idx][s];
                // 此时卡牌已被选定移入预留手牌，翻开对应牌堆顶补充金字塔市场对应槽位
                state.replenish_pyramid_slot(tier, s);
                (c, true)
            }
            None => {
                let tier_idx = tier.index();
                let c = state
                    .decks[tier_idx]
                    .pop()
                    .ok_or("Deck is empty for blind reserve")?;
                (c, false)
            }
        };

        state.players[current_player]
            .reserved_cards
            .push(crate::model::card::ReservedCard::new(card, is_public));

        Self::after_action_check(state);
        Ok(())
    }

    fn step_purchase_card(
        state: &mut GameState,
        from_reserved: bool,
        tier: CardTier,
        slot: usize,
    ) -> Result<(), String> {
        if state.phase != TurnPhase::MandatoryAction {
            return Err("Not in MandatoryAction phase".into());
        }
        let current_player = state.current_player;

        let card = if from_reserved {
            if slot >= state.players[current_player].reserved_cards.len() {
                return Err("Invalid reserved slot".into());
            }
            state.players[current_player].reserved_cards[slot].card
        } else {
            let tier_idx = tier.index();
            if slot >= state.pyramid[tier_idx].len() {
                return Err("Invalid pyramid slot".into());
            }
            state.pyramid[tier_idx][slot]
        };

        if card.color == CardColor::Joker && !state.players[current_player].has_any_bonus() {
            return Err("Cannot purchase Joker without existing bonuses".into());
        }

        let info = check_payment_divergence(&state.players[current_player], &card)
            .ok_or("Cannot afford card")?;

        // 若不存在战略支付分歧（无自由黄金或无可替代天然宝石），直接采用默认方案 0 步极速扣款结算
        if !info.has_divergence {
            let payment = compute_card_payment(&state.players[current_player], &card)
                .ok_or("Payment calculation failed")?;
            return Self::finalize_card_purchase(state, card, from_reserved, tier, slot, payment);
        }

        // 存在自由黄金与可替代宝石分歧，转入 Payment 阶段供玩家自主决策
        state.phase = TurnPhase::Payment {
            card,
            from_reserved,
            tier,
            slot,
            free_gold: info.free_gold,
            last_color_idx: 0,
            allocated_gold: [0; 6],
        };

        Ok(())
    }

    fn step_confirm_payment(state: &mut GameState) -> Result<(), String> {
        let (card, from_reserved, tier, slot, allocated_gold) = match &state.phase {
            TurnPhase::Payment {
                card,
                from_reserved,
                tier,
                slot,
                allocated_gold,
                ..
            } => (*card, *from_reserved, *tier, *slot, *allocated_gold),
            _ => return Err("Not in Payment phase".into()),
        };

        let current_player = state.current_player;
        let payment = compute_custom_payment(&state.players[current_player], &card, &allocated_gold)
            .ok_or("Invalid custom payment configuration")?;

        Self::finalize_card_purchase(state, card, from_reserved, tier, slot, payment)
    }

    fn step_pay_gold_for(state: &mut GameState, gem: GemType) -> Result<(), String> {
        let (card, from_reserved, tier, slot, mut free_gold, mut last_color_idx, mut allocated_gold) =
            match &state.phase {
                TurnPhase::Payment {
                    card,
                    from_reserved,
                    tier,
                    slot,
                    free_gold,
                    last_color_idx,
                    allocated_gold,
                } => (
                    *card,
                    *from_reserved,
                    *tier,
                    *slot,
                    *free_gold,
                    *last_color_idx,
                    *allocated_gold,
                ),
                _ => return Err("Not in Payment phase".into()),
            };

        if free_gold == 0 {
            return Err("No free gold available".into());
        }

        let idx = gem.index();
        if idx >= 6 {
            return Err("Cannot replace gold with gold".into());
        }
        if idx < last_color_idx {
            return Err("Colors must be selected in non-decreasing order".into());
        }

        let current_player = state.current_player;
        let info = check_payment_divergence(&state.players[current_player], &card)
            .ok_or("Cannot afford card")?;

        if allocated_gold[idx] >= info.max_replaceable[idx] {
            return Err("No more natural tokens of this color can be replaced".into());
        }

        allocated_gold[idx] += 1;
        free_gold -= 1;
        last_color_idx = idx;

        // 检查自动短路退出条件：
        // 1. 自由黄金耗尽 (free_gold == 0)；
        // 2. 或在单向保序约束下，后续区间内已无可替代天然宝石 ((last_color_idx..6).all(...))
        let no_more_eligible =
            (last_color_idx..6).all(|i| allocated_gold[i] >= info.max_replaceable[i]);
        if free_gold == 0 || no_more_eligible {
            let payment =
                compute_custom_payment(&state.players[current_player], &card, &allocated_gold)
                    .ok_or("Auto-settle payment calculation failed")?;
            Self::finalize_card_purchase(state, card, from_reserved, tier, slot, payment)
        } else {
            state.phase = TurnPhase::Payment {
                card,
                from_reserved,
                tier,
                slot,
                free_gold,
                last_color_idx,
                allocated_gold,
            };
            Ok(())
        }
    }

    /// 统一卡牌购买结算与进场结算
    fn finalize_card_purchase(
        state: &mut GameState,
        card: JewelCard,
        from_reserved: bool,
        tier: CardTier,
        slot: usize,
        payment: TokenCollection,
    ) -> Result<(), String> {
        let current_player = state.current_player;

        // 扣除支付标记并退回布袋
        state.players[current_player]
            .tokens
            .remove_collection(&payment)
            .map_err(|_| "Payment deduction failed")?;

        for gem in GemType::ALL {
            let cnt = payment.get(gem);
            for _ in 0..cnt {
                state.bag.push(gem);
            }
        }

        // 移除卡牌
        if from_reserved {
            if slot < state.players[current_player].reserved_cards.len() {
                state.players[current_player].reserved_cards.remove(slot);
            }
        } else {
            state.replenish_pyramid_slot(tier, slot);
        }

        // 卡牌能力与进场处理
        if card.color == CardColor::Joker {
            state.phase = TurnPhase::CardAbilityJoker { pending_card: card };
        } else {
            state.players[current_player].play_card(card, None);
            Self::handle_card_ability(state, card.ability, card.color);
        }

        Ok(())
    }

    fn step_assign_joker(state: &mut GameState, color: GemType) -> Result<(), String> {
        let (pending_card, ability) = match &state.phase {
            TurnPhase::CardAbilityJoker { pending_card } => (*pending_card, pending_card.ability),
            _ => return Err("Not in CardAbilityJoker phase".into()),
        };

        let current_player = state.current_player;
        if state.players[current_player].bonuses[color.index()] == 0 {
            return Err("Cannot assign Joker to color without bonus".into());
        }

        state.players[current_player].play_card(pending_card, Some(color));

        // 检查附带能力（如 3-11 号卡带 ExtraTurn）
        Self::handle_card_ability(state, ability, CardColor::from_gem_type(color).unwrap());
        Ok(())
    }

    fn step_take_same_color(state: &mut GameState, r: usize, c: usize) -> Result<(), String> {
        let target_color = match state.phase {
            TurnPhase::CardAbilitySameColor { color } => color,
            _ => return Err("Not in CardAbilitySameColor phase".into()),
        };

        let gem = state.board.get(r, c).ok_or("No token at position")?;
        if gem != target_color {
            return Err("Token does not match required color".into());
        }

        state.board.take(r, c);
        let current_player = state.current_player;
        state.players[current_player].tokens.add(gem, 1);

        Self::after_ability_check(state);
        Ok(())
    }

    fn step_steal_token(state: &mut GameState, gem: GemType) -> Result<(), String> {
        if state.phase != TurnPhase::CardAbilitySteal {
            return Err("Not in CardAbilitySteal phase".into());
        }
        if gem.is_gold() {
            return Err("Cannot steal gold".into());
        }

        let opponent = state.opponent_idx();
        let current_player = state.current_player;
        state.players[opponent]
            .tokens
            .remove(gem, 1)
            .map_err(|_| "Opponent does not have this token")?;
        state.players[current_player].tokens.add(gem, 1);

        Self::after_ability_check(state);
        Ok(())
    }

    fn step_select_royal(state: &mut GameState, royal_id: u8) -> Result<(), String> {
        if state.phase != TurnPhase::SelectRoyalCard {
            return Err("Not in SelectRoyalCard phase".into());
        }

        let royal_idx = state
            .royal_cards
            .iter()
            .position(|r| r.id == royal_id)
            .ok_or("Royal card not found on board")?;

        let royal = state.royal_cards.remove(royal_idx);
        let current_player = state.current_player;

        // 标记已消费的里程碑
        let milestones = state.players[current_player].pending_royal_milestones();
        if let Some(&m) = milestones.first() {
            state.players[current_player].mark_royal_claimed(m);
        }

        // 结算王室卡声望
        state.players[current_player].claim_royal_card(royal);

        // 结算王室卡能力
        match royal.ability {
            Some(RoyalAbility::ExtraTurn) => {
                state.extra_turn_granted = true;
                Self::after_ability_check(state);
            }
            Some(RoyalAbility::TakePrivilege) => {
                state.grant_privilege_to(current_player);
                Self::after_ability_check(state);
            }
            Some(RoyalAbility::StealToken) => {
                let opponent = state.opponent_idx();
                let has_stealable = GemType::BASIC_FIVE
                    .iter()
                    .any(|&g| state.players[opponent].tokens.get(g) > 0)
                    || state.players[opponent].tokens.get(GemType::Pearl) > 0;
                if has_stealable {
                    state.phase = TurnPhase::CardAbilitySteal;
                } else {
                    Self::after_ability_check(state);
                }
            }
            None => {
                Self::after_ability_check(state);
            }
        }

        Ok(())
    }

    fn step_discard_token(state: &mut GameState, gem: GemType) -> Result<(), String> {
        if state.phase != TurnPhase::DiscardTokens {
            return Err("Not in DiscardTokens phase".into());
        }
        let current_player = state.current_player;
        state.players[current_player]
            .tokens
            .remove(gem, 1)
            .map_err(|_| "Player does not have this token")?;
        state.bag.push(gem);

        if state.players[current_player].tokens.total() > 10 {
            state.phase = TurnPhase::DiscardTokens;
        } else {
            Self::finish_turn(state);
        }
        Ok(())
    }

    fn handle_card_ability(
        state: &mut GameState,
        ability: Option<CardAbility>,
        card_color: CardColor,
    ) {
        let current_player = state.current_player;
        match ability {
            Some(CardAbility::ExtraTurn) => {
                state.extra_turn_granted = true;
                Self::after_ability_check(state);
            }
            Some(CardAbility::TakePrivilege) => {
                state.grant_privilege_to(current_player);
                Self::after_ability_check(state);
            }
            Some(CardAbility::TakeSameColor) => {
                if let Some(target_color) = card_color.to_gem_type() {
                    let has_on_board = (0..5).any(|r| {
                        (0..5).any(|c| state.board.get(r, c) == Some(target_color))
                    });
                    if has_on_board {
                        state.phase = TurnPhase::CardAbilitySameColor { color: target_color };
                    } else {
                        Self::after_ability_check(state);
                    }
                } else {
                    Self::after_ability_check(state);
                }
            }
            Some(CardAbility::StealToken) => {
                let opponent = state.opponent_idx();
                let has_stealable = GemType::BASIC_FIVE
                    .iter()
                    .any(|&g| state.players[opponent].tokens.get(g) > 0)
                    || state.players[opponent].tokens.get(GemType::Pearl) > 0;
                if has_stealable {
                    state.phase = TurnPhase::CardAbilitySteal;
                } else {
                    Self::after_ability_check(state);
                }
            }
            Some(CardAbility::ColorCopyAndExtraTurn) => {
                state.extra_turn_granted = true;
                Self::after_ability_check(state);
            }
            Some(CardAbility::ColorCopy) | None => {
                Self::after_ability_check(state);
            }
        }
    }

    fn after_action_check(state: &mut GameState) {
        Self::after_ability_check(state);
    }

    fn after_ability_check(state: &mut GameState) {
        let current_player = state.current_player;

        // 检查王室卡是否达标 (3 顶或 6 顶王冠且场上有王室卡剩余)
        let milestones = state.players[current_player].pending_royal_milestones();
        if !milestones.is_empty() && !state.royal_cards.is_empty() {
            state.phase = TurnPhase::SelectRoyalCard;
            return;
        }

        // 检查手牌超限 (> 10)
        if state.players[current_player].tokens.total() > 10 {
            state.phase = TurnPhase::DiscardTokens;
            return;
        }

        // 完成回合
        Self::finish_turn(state);
    }

    fn finish_turn(state: &mut GameState) {
        let current_player = state.current_player;

        // 胜利条件检验
        if let Some(reason) = check_victory(&state.players[current_player]) {
            state.winner = Some((current_player, reason));
            state.phase = TurnPhase::GameOver(reason);
            return;
        }

        // 处理额外回合或轮转到对手
        if state.extra_turn_granted {
            state.extra_turn_granted = false;
        } else {
            state.current_player = 1 - state.current_player;
        }

        state.replenished_this_turn = false;
        state.privileges_used_this_turn = 0;
        state.turn_number += 1;
        state.phase = TurnPhase::OptionalActions;
    }
}
