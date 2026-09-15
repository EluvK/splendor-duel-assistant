use crate::model::card::JewelCard;
use crate::model::token::GemType;
use serde::{Deserialize, Serialize};

/// 获胜原因
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VictoryReason {
    TwentyPrestigePoints,
    TenCrowns,
    TenPointsSameColor(GemType),
}

/// 回合细粒度状态机阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnPhase {
    /// 可选行动阶段（可执行 0~2 项可选行动：使用特权卷轴、补充棋盘、或跳过进入强制行动）
    OptionalActions,

    /// 强制行动阶段（3 选 1：拿取连线标记、预留卡牌、购买卡牌）
    MandatoryAction,

    /// 变色复制卡（Joker）附着阶段：等待当前玩家选择附着到哪种已拥有 bonus 的基础颜色
    CardAbilityJoker { pending_card: JewelCard },

    /// 盘上取同色能力阶段：等待玩家指定棋盘上的 1 枚同色宝石
    CardAbilitySameColor { color: GemType },

    /// 偷取标记能力阶段：等待玩家指定从对手处偷取哪种非黄金标记
    CardAbilitySteal,

    /// 王室卡选择阶段：达到 3 或 6 顶王冠时，等待玩家从场上选择 1 张王室卡
    SelectRoyalCard,

    /// 手牌超限弃牌阶段：玩家标记总数超过 10 枚，等待逐枚选择弃置标记回布袋
    DiscardTokens,

    /// 预留卡牌黄金选择阶段：预留卡牌后，若盘上有黄金，等待玩家从棋盘选择 1 枚黄金
    SelectReserveGold,

    /// 游戏已结束，获胜者与胜利类型已确定
    GameOver(VictoryReason),
}
