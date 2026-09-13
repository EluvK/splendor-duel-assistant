use serde::{Deserialize, Serialize};

/// 璀璨宝石对决中的 7 种标记类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum GemType {
    White,  // 珍珠贝 / 钻石
    Blue,   // 蓝宝石
    Green,  // 绿宝石
    Red,    // 红宝石
    Black,  // 黑曜石
    Pearl,  // 珍珠 (稀缺，无卡牌奖励)
    Gold,   // 黄金 (万能，仅能通过预留获得)
}

impl<'de> Deserialize<'de> for GemType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "white" => Ok(GemType::White),
            "blue" => Ok(GemType::Blue),
            "green" => Ok(GemType::Green),
            "red" => Ok(GemType::Red),
            "black" => Ok(GemType::Black),
            "pearl" => Ok(GemType::Pearl),
            "gold" => Ok(GemType::Gold),
            _ => Err(serde::de::Error::custom(format!("unknown gem type: {s}"))),
        }
    }
}

impl GemType {
    pub const ALL: [GemType; 7] = [
        GemType::White,
        GemType::Blue,
        GemType::Green,
        GemType::Red,
        GemType::Black,
        GemType::Pearl,
        GemType::Gold,
    ];

    pub const BASIC_FIVE: [GemType; 5] = [
        GemType::White,
        GemType::Blue,
        GemType::Green,
        GemType::Red,
        GemType::Black,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        match self {
            GemType::White => 0,
            GemType::Blue => 1,
            GemType::Green => 2,
            GemType::Red => 3,
            GemType::Black => 4,
            GemType::Pearl => 5,
            GemType::Gold => 6,
        }
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(GemType::White),
            1 => Some(GemType::Blue),
            2 => Some(GemType::Green),
            3 => Some(GemType::Red),
            4 => Some(GemType::Black),
            5 => Some(GemType::Pearl),
            6 => Some(GemType::Gold),
            _ => None,
        }
    }

    #[inline]
    pub const fn is_gold(self) -> bool {
        matches!(self, GemType::Gold)
    }

    #[inline]
    pub const fn is_basic_gem(self) -> bool {
        matches!(
            self,
            GemType::White | GemType::Blue | GemType::Green | GemType::Red | GemType::Black
        )
    }
}

/// 标记数量集合（定长数组 [u8; 7]，下标对应 GemType::index）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenCollection {
    pub counts: [u8; 7],
}

impl TokenCollection {
    pub const EMPTY: Self = Self { counts: [0; 7] };

    pub const fn new() -> Self {
        Self { counts: [0; 7] }
    }

    /// 游戏开局总标记池（共 25 枚）
    pub const fn initial_bag() -> Self {
        Self {
            counts: [
                4, // White
                4, // Blue
                4, // Green
                4, // Red
                4, // Black
                2, // Pearl
                3, // Gold
            ],
        }
    }

    #[inline]
    pub fn get(&self, gem: GemType) -> u8 {
        self.counts[gem.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, gem: GemType) -> &mut u8 {
        &mut self.counts[gem.index()]
    }

    #[inline]
    pub fn set(&mut self, gem: GemType, val: u8) {
        self.counts[gem.index()] = val;
    }

    #[inline]
    pub fn add(&mut self, gem: GemType, count: u8) {
        self.counts[gem.index()] += count;
    }

    #[inline]
    pub fn remove(&mut self, gem: GemType, count: u8) -> Result<(), ()> {
        if self.counts[gem.index()] >= count {
            self.counts[gem.index()] -= count;
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline]
    pub fn total(&self) -> u8 {
        let mut sum = 0;
        let mut i = 0;
        while i < 7 {
            sum += self.counts[i];
            i += 1;
        }
        sum
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    pub fn add_collection(&mut self, other: &TokenCollection) {
        for i in 0..7 {
            self.counts[i] += other.counts[i];
        }
    }

    pub fn remove_collection(&mut self, other: &TokenCollection) -> Result<(), ()> {
        for i in 0..7 {
            if self.counts[i] < other.counts[i] {
                return Err(());
            }
        }
        for i in 0..7 {
            self.counts[i] -= other.counts[i];
        }
        Ok(())
    }
}
