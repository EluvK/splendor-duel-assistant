use super::card::CardTier;
use super::token::GemType;
use serde::{Deserialize, Serialize};

/// 游戏动作枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// 可选行动：跳过剩余可选行动，进入强制行动
    SkipOptional,

    /// 可选行动：使用 1 个特权卷轴从棋盘指定坐标拿取 1 枚非黄金标记
    UsePrivilege { r: usize, c: usize },

    /// 可选行动：补充棋盘（沿螺旋轨道从布袋填满空格，对手获得 1 特权）
    ReplenishBoard,

    /// 强制行动：拿取 1 至 3 枚连线相邻非黄金标记
    TakeTokens {
        count: u8,
        positions: [(usize, usize); 3],
    },

    /// 强制行动：拿 1 枚黄金（若盘上有）并预留 1 张珠宝卡
    /// - 若 slot 为 Some(i)：预留金字塔中 tier 等级第 i 个明牌槽位（0..market_len）
    /// - 若 slot 为 None：从 tier 等级牌堆顶端暗抽 1 张预留
    ReserveCard {
        tier: CardTier,
        slot: Option<usize>,
    },

    /// 强制行动：购买 1 张珠宝卡
    /// - from_reserved 为 false 时：购买金字塔中 tier 等级第 slot 个槽位的明牌
    /// - from_reserved 为 true 时：购买玩家自己手牌区第 slot 张预留卡（0..reserved_len）
    PurchaseCard {
        from_reserved: bool,
        tier: CardTier,
        slot: usize,
    },

    /// 预留卡牌连锁动作：从棋盘指定坐标 (r, c) 拿取 1 枚黄金
    TakeGoldToken { r: usize, c: usize },

    /// 卡牌连锁能力：变色复制卡（Joker）选择附着的宝石颜色（必须是已有 bonus 颜色）
    AssignJokerColor { color: GemType },

    /// 卡牌连锁能力：从棋盘指定位置拿取 1 枚与本卡同色的宝石
    TakeSameColorToken { r: usize, c: usize },

    /// 卡牌/王室能力：从对手处偷取 1 枚非黄金标记
    StealToken { gem: GemType },

    /// 王室卡选择：王冠达到 3 顶或 6 顶时，选择场上可用的王室卡 (royal_id 0..4)
    SelectRoyal { royal_id: u8 },

    /// 手牌超限弃牌：手中标记超过 10 枚时，自选弃置 1 枚标记回布袋
    DiscardToken { gem: GemType },

    /// 支付自主决策：确认当前支付方案并结算扣除
    ConfirmPayment,

    /// 支付自主决策：使用 1 枚自由黄金替代 1 枚指定颜色宝石（W, B, G, R, K, Pearl）
    PayGoldFor { gem: GemType },
}
