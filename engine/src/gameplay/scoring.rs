use crate::game_state::phase::VictoryReason;
use crate::game_state::player::PlayerState;
use crate::model::token::GemType;

/// 检查玩家是否达成任一终局胜利条件
pub fn check_victory(player: &PlayerState) -> Option<VictoryReason> {
    // 1. 总声望达到 20 分
    if player.total_points >= 20 {
        return Some(VictoryReason::TwentyPrestigePoints);
    }

    // 2. 累计王冠达到 10 顶
    if player.total_crowns >= 10 {
        return Some(VictoryReason::TenCrowns);
    }

    // 3. 某种单色珠宝卡声望达到 10 分
    for gem in GemType::BASIC_FIVE {
        if player.color_points[gem.index()] >= 10 {
            return Some(VictoryReason::TenPointsSameColor(gem));
        }
    }

    None
}
