use super::token::GemType;
use serde::{Deserialize, Serialize};

/// 珠宝卡等级（1 级、2 级、3 级）
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CardTier {
    Tier1 = 0,
    Tier2 = 1,
    Tier3 = 2,
}

impl CardTier {
    pub const ALL: [CardTier; 3] = [CardTier::Tier1, CardTier::Tier2, CardTier::Tier3];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn market_capacity(self) -> usize {
        match self {
            CardTier::Tier1 => 5,
            CardTier::Tier2 => 4,
            CardTier::Tier3 => 3,
        }
    }
}

/// 珠宝卡颜色类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardColor {
    White,
    Blue,
    Green,
    Red,
    Black,
    Points, // 纯声望分卡牌（0 宝石 bonus）
    Joker,  // 变色万能卡（购买时需附着已有颜色）
}

impl CardColor {
    #[inline]
    pub fn to_gem_type(self) -> Option<GemType> {
        match self {
            CardColor::White => Some(GemType::White),
            CardColor::Blue => Some(GemType::Blue),
            CardColor::Green => Some(GemType::Green),
            CardColor::Red => Some(GemType::Red),
            CardColor::Black => Some(GemType::Black),
            CardColor::Points | CardColor::Joker => None,
        }
    }

    #[inline]
    pub fn from_gem_type(gem: GemType) -> Option<Self> {
        match gem {
            GemType::White => Some(CardColor::White),
            GemType::Blue => Some(CardColor::Blue),
            GemType::Green => Some(CardColor::Green),
            GemType::Red => Some(CardColor::Red),
            GemType::Black => Some(CardColor::Black),
            GemType::Pearl | GemType::Gold => None,
        }
    }
}

/// 卡牌能力
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardAbility {
    ExtraTurn,              // 额外回合
    TakePrivilege,          // 拿取 1 特权卷轴
    TakeSameColor,          // 盘上取同色
    StealToken,             // 偷取对手 1 枚非黄金标记
    ColorCopy,              // 变色复制
    ColorCopyAndExtraTurn,  // 变色复制 + 额外回合 (3-11 号卡)
}

/// 卡牌成本 (白、蓝、绿、红、黑、珍珠)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CardCost {
    pub white: u8,
    pub blue: u8,
    pub green: u8,
    pub red: u8,
    pub black: u8,
    pub pearl: u8,
}

impl CardCost {
    pub const fn new(white: u8, blue: u8, green: u8, red: u8, black: u8, pearl: u8) -> Self {
        Self {
            white,
            blue,
            green,
            red,
            black,
            pearl,
        }
    }

    #[inline]
    pub const fn get(&self, gem: GemType) -> u8 {
        match gem {
            GemType::White => self.white,
            GemType::Blue => self.blue,
            GemType::Green => self.green,
            GemType::Red => self.red,
            GemType::Black => self.black,
            GemType::Pearl => self.pearl,
            GemType::Gold => 0,
        }
    }
}

/// 珠宝卡定义
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JewelCard {
    pub id: u8, // 0..67
    pub tier: CardTier,
    pub color: CardColor,
    pub points: u8,
    pub bonus: u8,
    pub ability: Option<CardAbility>,
    pub crowns: u8,
    pub cost: CardCost,
}

/// 王室卡能力
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoyalAbility {
    StealToken,    // 偷取非黄金标记
    TakePrivilege, // 拿取特权卷轴
    ExtraTurn,     // 额外回合
}

/// 王室卡定义
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoyalCard {
    pub id: u8, // 0..4
    pub points: u8,
    pub ability: Option<RoyalAbility>,
}
