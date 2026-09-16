use crate::game_state::phase::TurnPhase;
use crate::game_state::state::GameState;
use crate::gameplay::payment::check_payment_divergence;
use crate::model::action::Action;
use crate::model::card::{CardColor, CardTier, JewelCard};
use crate::model::token::GemType;

/// 规则引擎：根据当前游戏状态与阶段生成所有合法动作
pub struct RuleEngine;

impl RuleEngine {
    pub fn legal_actions(state: &GameState) -> Vec<Action> {
        match &state.phase {
            TurnPhase::OptionalActions => Self::legal_optional_actions(state),
            TurnPhase::MandatoryAction => Self::legal_mandatory_actions(state),
            TurnPhase::SelectReserveCard => Self::legal_reserve_card_actions(state),
            TurnPhase::Payment {
                card,
                free_gold,
                last_color_idx,
                allocated_gold,
                ..
            } => Self::legal_payment_actions(
                state,
                card,
                *free_gold,
                *last_color_idx,
                allocated_gold,
            ),
            TurnPhase::CardAbilityJoker { .. } => Self::legal_joker_actions(state),
            TurnPhase::CardAbilitySameColor { color } => {
                Self::legal_same_color_actions(state, *color)
            }
            TurnPhase::CardAbilitySteal => Self::legal_steal_actions(state),
            TurnPhase::SelectRoyalCard => Self::legal_royal_actions(state),
            TurnPhase::DiscardTokens => Self::legal_discard_actions(state),
            TurnPhase::GameOver(_) => Vec::new(),
        }
    }

    /// 极速判定当前玩家是否存在任意合法的强制行动（Mandatory Actions）
    /// 99.99% 的情况下第一行 `board.has_non_gold()` 就会直接短路返回 true（耗时仅 1~2 个 CPU 周期），
    /// 绝不进行卡牌遍历与多余计算，完全零性能损耗。
    pub fn has_any_mandatory_action(state: &GameState) -> bool {
        // 1. 最强短路：盘上只要有任意 1 枚非黄金标记，强制行动必定合法（可拿取该 1 枚标记）
        if state.board.has_non_gold() {
            return true;
        }

        let player = &state.players[state.current_player];

        // 2. 次级短路：盘上有黄金且手牌预留未满 3 张，可执行预留行动
        if state.board.has_gold() && player.reserved_cards.len() < 3 {
            return true;
        }

        // 3. 仅在盘上完全没有非黄金标记且无法预留的极罕见枯竭局面下，检查能否购买卡牌
        for tier in CardTier::ALL {
            for card in &state.pyramid[tier.index()] {
                if card.color == CardColor::Joker && !player.has_any_bonus() {
                    continue;
                }
                if player.can_afford(card) {
                    return true;
                }
            }
        }
        for card in &player.reserved_cards {
            if card.card.color == CardColor::Joker && !player.has_any_bonus() {
                continue;
            }
            if player.can_afford(&card.card) {
                return true;
            }
        }

        false
    }

    fn legal_optional_actions(state: &GameState) -> Vec<Action> {
        let player = &state.players[state.current_player];
        let mut actions = Vec::with_capacity(32);

        let can_mandatory = Self::has_any_mandatory_action(state);
        let can_replenish = !state.replenished_this_turn && !state.bag.is_empty() && state.board.has_empty_slot();

        // 1. 跳过可选行动，直接进入强制行动：
        // 规则书规定：若本回合没有任何必选操作可以做，则可选操作中必须补充棋盘，不可跳过！
        // 仅在后续存在合法强制行动、或连补盘都不可能的死锁极端情况下，才允许跳过。
        if can_mandatory || !can_replenish {
            actions.push(Action::SkipOptional);
        }

        // 2. 使用特权卷轴（必须持有特权卷轴，且本回合尚未补盘）
        // 规则约束：必须先使用特权卷轴，再补充棋盘；一旦补盘，本回合不可再使用特权
        if !state.replenished_this_turn && player.privileges > 0 {
            for r in 0..5 {
                for c in 0..5 {
                    if let Some(gem) = state.board.get(r, c) {
                        if !gem.is_gold() {
                            actions.push(Action::UsePrivilege { r, c });
                        }
                    }
                }
            }
        }

        // 3. 补充棋盘（前提：本回合尚未补充过棋盘，且布袋非空、棋盘有空格）
        if can_replenish {
            actions.push(Action::ReplenishBoard);
        }

        actions
    }

    fn legal_mandatory_actions(state: &GameState) -> Vec<Action> {
        let player = &state.players[state.current_player];
        let mut actions = Vec::with_capacity(128);

        // 选项 A：拿取 1 至 3 枚连线相邻非黄金标记
        let lines = state.board.find_all_lines();
        for line in lines {
            actions.push(Action::TakeTokens {
                count: line.count,
                positions: line.positions,
            });
        }

        // 选项 B：拿 1 枚黄金并开启预留流程（前提：棋盘上必须至少有 1 枚黄金，且预留手牌未达上限 3 张）
        if state.board.has_gold() && player.reserved_cards.len() < 3 {
            for r in 0..5 {
                for c in 0..5 {
                    if state.board.get(r, c) == Some(GemType::Gold) {
                        actions.push(Action::TakeGoldToken { r, c });
                    }
                }
            }
        }

        // 选项 C：购买卡牌
        // 1. 从金字塔明牌购买
        for tier in CardTier::ALL {
            for (slot, card) in state.pyramid[tier.index()].iter().enumerate() {
                // 变色卡必须在拥有至少 1 种已有 bonus 时才可购买
                if card.color == CardColor::Joker && !player.has_any_bonus() {
                    continue;
                }
                if player.can_afford(card) {
                    actions.push(Action::PurchaseCard {
                        from_reserved: false,
                        tier,
                        slot,
                    });
                }
            }
        }

        // 2. 从自己预留卡购买
        for (slot, card) in player.reserved_cards.iter().enumerate() {
            if card.color == CardColor::Joker && !player.has_any_bonus() {
                continue;
            }
            if player.can_afford(card) {
                actions.push(Action::PurchaseCard {
                    from_reserved: true,
                    tier: card.tier,
                    slot,
                });
            }
        }

        // 边界保护：若没有任何合法强制行动（极端情况）
        if actions.is_empty() {
            if !state.bag.is_empty() && state.board.has_empty_slot() {
                actions.push(Action::ReplenishBoard);
            } else {
                // 若连补板都不可行（盘上无非黄金、袋空、预留满、买不起），允许跳过强制行动以防死锁
                actions.push(Action::SkipOptional);
            }
        }

        actions
    }

    fn legal_joker_actions(state: &GameState) -> Vec<Action> {
        let player = &state.players[state.current_player];
        let mut actions = Vec::with_capacity(5);
        for gem in GemType::BASIC_FIVE {
            if player.bonuses[gem.index()] > 0 {
                actions.push(Action::AssignJokerColor { color: gem });
            }
        }
        actions
    }

    fn legal_same_color_actions(state: &GameState, target_color: GemType) -> Vec<Action> {
        let mut actions = Vec::with_capacity(8);
        for r in 0..5 {
            for c in 0..5 {
                if state.board.get(r, c) == Some(target_color) {
                    actions.push(Action::TakeSameColorToken { r, c });
                }
            }
        }
        actions
    }

    fn legal_steal_actions(state: &GameState) -> Vec<Action> {
        let opponent = &state.players[state.opponent_idx()];
        let mut actions = Vec::with_capacity(6);
        for &gem in GemType::BASIC_FIVE.iter() {
            if opponent.tokens.get(gem) > 0 {
                actions.push(Action::StealToken { gem });
            }
        }
        if opponent.tokens.get(GemType::Pearl) > 0 {
            actions.push(Action::StealToken {
                gem: GemType::Pearl,
            });
        }
        actions
    }

    fn legal_royal_actions(state: &GameState) -> Vec<Action> {
        state
            .royal_cards
            .iter()
            .map(|r| Action::SelectRoyal { royal_id: r.id })
            .collect()
    }

    fn legal_discard_actions(state: &GameState) -> Vec<Action> {
        let player = &state.players[state.current_player];
        let mut actions = Vec::with_capacity(7);
        for gem in GemType::ALL {
            if player.tokens.get(gem) > 0 {
                actions.push(Action::DiscardToken { gem });
            }
        }
        actions
    }

    fn legal_reserve_card_actions(state: &GameState) -> Vec<Action> {
        let mut actions = Vec::with_capacity(16);
        for tier in CardTier::ALL {
            // 金字塔明牌
            for slot in 0..state.pyramid[tier.index()].len() {
                actions.push(Action::ReserveCard {
                    tier,
                    slot: Some(slot),
                });
            }
            // 牌堆顶盲抽
            if !state.decks[tier.index()].is_empty() {
                actions.push(Action::ReserveCard { tier, slot: None });
            }
        }
        actions
    }

    fn legal_payment_actions(
        state: &GameState,
        card: &JewelCard,
        free_gold: u8,
        last_color_idx: usize,
        allocated_gold: &[u8; 6],
    ) -> Vec<Action> {
        let player = &state.players[state.current_player];
        let mut actions = Vec::with_capacity(7);

        // 1. 确认当前方案并结算退出始终合法
        actions.push(Action::ConfirmPayment);

        // 2. 若仍有自由黄金，且对应颜色仍有天然宝石可被替代，且满足单向保序 (idx >= last_color_idx)
        if free_gold > 0 {
            if let Some(info) = check_payment_divergence(player, card) {
                for gem in GemType::ALL {
                    let idx = gem.index();
                    if idx < 6 && idx >= last_color_idx {
                        if allocated_gold[idx] < info.max_replaceable[idx] {
                            actions.push(Action::PayGoldFor { gem });
                        }
                    }
                }
            }
        }

        actions
    }
}
