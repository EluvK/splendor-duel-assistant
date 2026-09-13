use crate::game_state::player::PlayerState;
use crate::model::card::JewelCard;
use crate::model::token::{GemType, TokenCollection};

/// 计算购买卡牌所需支付的标记集合（优先使用自身折扣，再用真实标记，最后用黄金填补缺口）
pub fn compute_card_payment(player: &PlayerState, card: &JewelCard) -> Option<TokenCollection> {
    let mut payment = TokenCollection::new();
    let mut gold_needed = 0;

    // 1. 计算 5 种基础宝石支付
    for gem in GemType::BASIC_FIVE {
        let cost = card.cost.get(gem);
        let discount = player.get_bonus(gem);
        let required = cost.saturating_sub(discount);
        let have = player.tokens.get(gem);

        let pay_exact = have.min(required);
        payment.set(gem, pay_exact);

        if required > pay_exact {
            gold_needed += required - pay_exact;
        }
    }

    // 2. 计算珍珠支付（无折扣）
    let pearl_req = card.cost.pearl;
    let pearl_have = player.tokens.get(GemType::Pearl);
    let pay_pearl = pearl_have.min(pearl_req);
    payment.set(GemType::Pearl, pay_pearl);
    if pearl_req > pay_pearl {
        gold_needed += pearl_req - pay_pearl;
    }

    // 3. 检查黄金是否足够填补缺口
    if player.tokens.get(GemType::Gold) < gold_needed {
        return None;
    }

    payment.set(GemType::Gold, gold_needed);
    Some(payment)
}
