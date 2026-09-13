use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::board::Board;
use super::phase::{TurnPhase, VictoryReason};
use super::player::PlayerState;
use crate::model::card::{CardTier, JewelCard, RoyalCard};
use crate::model::data::{ALL_JEWEL_CARDS, ALL_ROYAL_CARDS};
use crate::model::token::GemType;

/// 完整游戏状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub bag: Vec<GemType>,
    pub decks: [Vec<JewelCard>; 3],
    pub pyramid: [Vec<JewelCard>; 3],
    pub royal_cards: Vec<RoyalCard>,
    pub privilege_pool: u8,
    pub players: [PlayerState; 2],
    pub current_player: usize,
    pub phase: TurnPhase,
    pub turn_number: u32,
    pub extra_turn_granted: bool,
    pub winner: Option<(usize, VictoryReason)>,
}

impl GameState {
    /// 依据种子初始化一局标准的《璀璨宝石：对决》
    pub fn new_game(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // 1. 分别洗牌 3 个等级的珠宝卡
        let mut decks = [Vec::new(), Vec::new(), Vec::new()];
        for card in ALL_JEWEL_CARDS.iter() {
            decks[card.tier.index()].push(*card);
        }
        for deck in decks.iter_mut() {
            deck.shuffle(&mut rng);
        }

        // 2. 翻开金字塔明牌
        let mut pyramid = [Vec::new(), Vec::new(), Vec::new()];
        for tier in CardTier::ALL {
            let cap = tier.market_capacity();
            for _ in 0..cap {
                if let Some(card) = decks[tier.index()].pop() {
                    pyramid[tier.index()].push(card);
                }
            }
        }

        // 3. 将全部 25 枚标记放入布袋并洗匀
        let mut bag = Vec::with_capacity(25);
        for &gem in GemType::BASIC_FIVE.iter() {
            for _ in 0..4 {
                bag.push(gem);
            }
        }
        for _ in 0..2 {
            bag.push(GemType::Pearl);
        }
        for _ in 0..3 {
            bag.push(GemType::Gold);
        }
        bag.shuffle(&mut rng);

        // 4. 沿螺旋轨道摸出 25 枚填满棋盘，此时布袋变空
        let mut board = Board::new();
        board.fill_spiral(&mut bag);

        // 5. 场上放置 4 张王室卡
        let royal_cards = ALL_ROYAL_CARDS.to_vec();

        // 6. 后手玩家（Player 1）开局直接获得 1 个特权卷轴，公用池留 2 个
        let p0 = PlayerState::new(0);
        let mut p1 = PlayerState::new(1);
        p1.privileges = 1;

        Self {
            board,
            bag,
            decks,
            pyramid,
            royal_cards,
            privilege_pool: 2,
            players: [p0, p1],
            current_player: 0,
            phase: TurnPhase::OptionalActions,
            turn_number: 1,
            extra_turn_granted: false,
            winner: None,
        }
    }

    /// 特权流转：优先从公用池取；公用池空时从对手处偷取；若己方已有 3 个则不再获得
    pub fn grant_privilege_to(&mut self, player_idx: usize) {
        if self.players[player_idx].privileges >= 3 {
            return;
        }

        if self.privilege_pool > 0 {
            self.privilege_pool -= 1;
            self.players[player_idx].privileges += 1;
        } else {
            let opponent_idx = 1 - player_idx;
            if self.players[opponent_idx].privileges > 0 {
                self.players[opponent_idx].privileges -= 1;
                self.players[player_idx].privileges += 1;
            }
        }
    }

    /// 归还特权卷轴至棋盘公用池
    pub fn return_privilege_from(&mut self, player_idx: usize) {
        if self.players[player_idx].privileges > 0 {
            self.players[player_idx].privileges -= 1;
            self.privilege_pool += 1;
        }
    }

    /// 金字塔空槽补牌
    pub fn replenish_pyramid_slot(&mut self, tier: CardTier, slot: usize) {
        if slot < self.pyramid[tier.index()].len() {
            if let Some(replacement) = self.decks[tier.index()].pop() {
                self.pyramid[tier.index()][slot] = replacement;
            } else {
                self.pyramid[tier.index()].remove(slot);
            }
        }
    }

    /// 对手玩家索引
    #[inline]
    pub fn opponent_idx(&self) -> usize {
        1 - self.current_player
    }
}
