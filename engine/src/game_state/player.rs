use crate::model::card::{CardColor, JewelCard, ReservedCard, RoyalCard};
use crate::model::token::{GemType, TokenCollection};
use serde::{Deserialize, Serialize};

/// 玩家状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerState {
    pub id: usize, // 0 或 1
    pub tokens: TokenCollection,
    pub cards: Vec<JewelCard>,
    pub reserved_cards: Vec<ReservedCard>, // 上限 3 张
    pub royal_cards: Vec<RoyalCard>,
    pub privileges: u8, // 上限 3 个

    // 缓存计算属性加速规则判定
    pub bonuses: [u8; 5],      // 白、蓝、绿、红、黑永久折抵
    pub color_points: [u8; 5], // 5 种基础颜色卡牌上的声望
    pub points_card_points: u8,// 纯分卡声望
    pub total_points: u8,
    pub total_crowns: u8,

    // 王室卡触发记录：[0] 对应 3 顶王冠，[1] 对应 6 顶王冠
    pub royals_claimed: [bool; 2],
}

impl PlayerState {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            tokens: TokenCollection::new(),
            cards: Vec::with_capacity(30),
            reserved_cards: Vec::with_capacity(3),
            royal_cards: Vec::with_capacity(2),
            privileges: 0,
            bonuses: [0; 5],
            color_points: [0; 5],
            points_card_points: 0,
            total_points: 0,
            total_crowns: 0,
            royals_claimed: [false; 2],
        }
    }

    /// 玩家是否拥有至少 1 个基础宝石永久 bonus（用于判定是否允许购买变色 Joker 卡）
    #[inline]
    pub fn has_any_bonus(&self) -> bool {
        self.bonuses.iter().any(|&b| b > 0)
    }

    /// 获取特定基础宝石的当前永久折抵
    #[inline]
    pub fn get_bonus(&self, gem: GemType) -> u8 {
        if gem.is_basic_gem() {
            self.bonuses[gem.index()]
        } else {
            0
        }
    }

    /// 检查并返回当前达标但尚未领取的王室卡里程碑列表（3 或 6）
    pub fn pending_royal_milestones(&self) -> Vec<u8> {
        let mut milestones = Vec::with_capacity(2);
        if self.total_crowns >= 3 && !self.royals_claimed[0] {
            milestones.push(3);
        }
        if self.total_crowns >= 6 && !self.royals_claimed[1] {
            milestones.push(6);
        }
        milestones
    }

    /// 标记已领取对应王冠里程碑的王室卡
    pub fn mark_royal_claimed(&mut self, milestone: u8) {
        if milestone == 3 {
            self.royals_claimed[0] = true;
        } else if milestone == 6 {
            self.royals_claimed[1] = true;
        }
    }

    /// 添加打出的珠宝卡并结算其属性与附着颜色
    pub fn play_card(&mut self, card: JewelCard, attached_color: Option<GemType>) {
        self.cards.push(card);
        self.total_points += card.points;
        self.total_crowns += card.crowns;

        match card.color {
            CardColor::White | CardColor::Blue | CardColor::Green | CardColor::Red | CardColor::Black => {
                let idx = card.color.to_gem_type().unwrap().index();
                self.bonuses[idx] += card.bonus;
                self.color_points[idx] += card.points;
            }
            CardColor::Points => {
                self.points_card_points += card.points;
            }
            CardColor::Joker => {
                if let Some(color) = attached_color {
                    let idx = color.index();
                    self.bonuses[idx] += card.bonus;
                    self.color_points[idx] += card.points;
                }
            }
        }
    }

    /// 添加已获得的王室卡
    pub fn claim_royal_card(&mut self, royal: RoyalCard) {
        self.total_points += royal.points;
        self.royal_cards.push(royal);
    }

    /// 判定是否能够支付某张卡牌（扣除 bonus 折抵后，用真实标记和黄金是否足够代偿）
    pub fn can_afford(&self, card: &JewelCard) -> bool {
        let mut gold_needed: u8 = 0;

        for gem in GemType::BASIC_FIVE {
            let cost = card.cost.get(gem);
            let discount = self.get_bonus(gem);
            let required = cost.saturating_sub(discount);
            let have = self.tokens.get(gem);
            if have < required {
                gold_needed += required - have;
            }
        }

        // 珍珠无法被 bonus 折抵
        let pearl_req = card.cost.pearl;
        let pearl_have = self.tokens.get(GemType::Pearl);
        if pearl_have < pearl_req {
            gold_needed += pearl_req - pearl_have;
        }

        self.tokens.get(GemType::Gold) >= gold_needed
    }
}
