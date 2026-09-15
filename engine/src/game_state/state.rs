use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::board::Board;
use super::phase::{TurnPhase, VictoryReason};
use super::player::PlayerState;
use crate::model::card::{CardTier, JewelCard, RoyalCard};
use crate::model::data::{ALL_JEWEL_CARDS, ALL_ROYAL_CARDS};
use crate::model::stack_vec::StackVec;
use crate::model::token::GemType;

pub const MAX_BAG_CAPACITY: usize = 25;
pub const MAX_DECK_CAPACITY: usize = 30;
pub const MAX_PYRAMID_CAPACITY: usize = 5;
pub const MAX_ROYAL_CARDS: usize = 4;

/// 完整游戏状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub bag: StackVec<GemType, MAX_BAG_CAPACITY>,
    pub decks: [StackVec<JewelCard, MAX_DECK_CAPACITY>; 3],
    pub pyramid: [StackVec<JewelCard, MAX_PYRAMID_CAPACITY>; 3],
    pub royal_cards: StackVec<RoyalCard, MAX_ROYAL_CARDS>,
    pub privilege_pool: u8,
    pub players: [PlayerState; 2],
    pub current_player: usize,
    pub phase: TurnPhase,
    pub turn_number: u32,
    pub extra_turn_granted: bool,
    pub winner: Option<(usize, VictoryReason)>,
    pub replenished_this_turn: bool,
    pub privileges_used_this_turn: u8,
    pub rng_seed: u64,
    pub rng_counter: u64,
}

impl GameState {
    /// 依据种子初始化一局标准的《璀璨宝石：对决》
    pub fn new_game(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // 1. 分别洗牌 3 个等级的珠宝卡
        let mut decks = [StackVec::new(), StackVec::new(), StackVec::new()];
        for card in ALL_JEWEL_CARDS.iter() {
            decks[card.tier.index()].push(*card);
        }
        for deck in decks.iter_mut() {
            deck.shuffle(&mut rng);
        }

        // 2. 翻开金字塔明牌
        let mut pyramid = [StackVec::new(), StackVec::new(), StackVec::new()];
        for tier in CardTier::ALL {
            let cap = tier.market_capacity();
            for _ in 0..cap {
                if let Some(card) = decks[tier.index()].pop() {
                    pyramid[tier.index()].push(card);
                }
            }
        }

        // 3. 将全部 25 枚标记放入布袋并洗匀
        let mut bag = StackVec::new();
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
        let mut royal_cards = StackVec::new();
        royal_cards.extend(ALL_ROYAL_CARDS.iter().copied());

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
            replenished_this_turn: false,
            privileges_used_this_turn: 0,
            rng_seed: seed,
            rng_counter: 0,
        }
    }

    /// 步进并衍生下一个高质量伪随机种子 (基于 SplitMix64 算法)
    #[inline]
    pub fn next_rng_seed(&mut self) -> u64 {
        self.rng_counter = self.rng_counter.wrapping_add(1);
        let mut z = self.rng_seed.wrapping_add(self.rng_counter.wrapping_mul(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// 针对指定观察者视角执行确定化重抽样（Determinization / POMDP 信念状态展开）
    ///
    /// 将对手盲抽暗牌与剩余牌堆卡牌统一归入未知卡牌池，按等级洗匀后重新发给对手暗牌与重构剩余牌堆。
    /// 观察者自身的全部卡牌、金字塔明牌、双方已购卡、对手公开预留卡均严格保持不变，卡牌总数严格满足守恒律。
    pub fn determinize_for_player<R: Rng + ?Sized>(&self, observer: usize, rng: &mut R) -> Self {
        let mut sim_state = self.clone();
        let opp = 1 - observer;

        // 1. 统计观察者已知的所有卡牌 ID (0..67)
        let mut known_card_ids = [false; 68];
        for row in &self.pyramid {
            for c in row {
                known_card_ids[c.id as usize] = true;
            }
        }
        for p in &self.players {
            for c in &p.cards {
                known_card_ids[c.id as usize] = true;
            }
        }
        for rc in &self.players[observer].reserved_cards {
            known_card_ids[rc.card.id as usize] = true;
        }
        for rc in &self.players[opp].reserved_cards {
            if rc.is_public {
                known_card_ids[rc.card.id as usize] = true;
            }
        }

        // 2. 按 Tier 搜集未知卡牌并重抽样 (栈上固定容量缓冲，零堆内存分配)
        for tier in CardTier::ALL {
            let t_idx = tier.index();
            // 各 Tier 卡牌数量上限：Tier1 为 30 张，Tier2 为 24 张，Tier3 为 13 张
            let mut unseen_buf = [ALL_JEWEL_CARDS[0]; 32];
            let mut count = 0;
            for c in ALL_JEWEL_CARDS.iter() {
                if c.tier == tier && !known_card_ids[c.id as usize] {
                    unseen_buf[count] = *c;
                    count += 1;
                }
            }

            unseen_buf[..count].shuffle(rng);

            // 优先替换对手该 Tier 的盲抽暗牌
            let mut pop_idx = count;
            for rc in sim_state.players[opp].reserved_cards.iter_mut() {
                if !rc.is_public && rc.card.tier == tier && pop_idx > 0 {
                    pop_idx -= 1;
                    rc.card = unseen_buf[pop_idx];
                }
            }

            // 剩余未知卡复用已有 ArrayVec (clear + extend，零堆分配)
            sim_state.decks[t_idx].clear();
            sim_state.decks[t_idx].extend(unseen_buf[..pop_idx].iter().copied());
        }

        sim_state
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

    /// 当前轮数 (Round，单玩家完整行动轮次，与 turn_number 等价)
    #[inline]
    pub fn round_number(&self) -> u32 {
        self.turn_number
    }
}
