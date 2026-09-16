use crate::game_state::player::PlayerState;
use crate::model::card::JewelCard;
use crate::model::token::{GemType, TokenCollection};

/// 支付分歧检测信息
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaymentDivergenceInfo {
    /// 是否存在战略支付分歧（存在自由黄金且存在可被替代的天然宝石）
    pub has_divergence: bool,
    /// 强制用于补齐宝石缺口的黄金数
    pub mandatory_gold: u8,
    /// 可由玩家自由支配用于替代天然宝石的黄金数
    pub free_gold: u8,
    /// 6 种资源（5 基础宝石 + 珍珠）默认情况下分别需要支付的天然标记数（即可被自由黄金替代的上限）
    pub max_replaceable: [u8; 6],
}

/// 分析卡牌购买时的支付需求、强制黄金与自由黄金分歧
pub fn check_payment_divergence(
    player: &PlayerState,
    card: &JewelCard,
) -> Option<PaymentDivergenceInfo> {
    let mut mandatory_gold = 0u8;
    let mut max_replaceable = [0u8; 6];

    // 1. 5 种基础宝石
    for gem in GemType::BASIC_FIVE {
        let cost = card.cost.get(gem);
        let discount = player.get_bonus(gem);
        let required = cost.saturating_sub(discount);
        let have = player.tokens.get(gem);

        let pay_exact = have.min(required);
        max_replaceable[gem.index()] = pay_exact;

        if required > pay_exact {
            mandatory_gold += required - pay_exact;
        }
    }

    // 2. 珍珠
    let pearl_req = card.cost.pearl;
    let pearl_have = player.tokens.get(GemType::Pearl);
    let pay_pearl = pearl_have.min(pearl_req);
    max_replaceable[GemType::Pearl.index()] = pay_pearl;

    if pearl_req > pay_pearl {
        mandatory_gold += pearl_req - pay_pearl;
    }

    let gold_have = player.tokens.get(GemType::Gold);
    if gold_have < mandatory_gold {
        return None; // 黄金不足以补齐缺口，买不起
    }

    let free_gold = gold_have - mandatory_gold;
    let any_replaceable = max_replaceable.iter().any(|&cnt| cnt > 0);
    let has_divergence = free_gold > 0 && any_replaceable;

    Some(PaymentDivergenceInfo {
        has_divergence,
        mandatory_gold,
        free_gold,
        max_replaceable,
    })
}

/// 计算购买卡牌所需支付的标记集合（默认最优方案：优先使用自身折扣，再用天然标记，最后用黄金填补缺口）
pub fn compute_card_payment(player: &PlayerState, card: &JewelCard) -> Option<TokenCollection> {
    let info = check_payment_divergence(player, card)?;
    let mut payment = TokenCollection::new();

    for gem in GemType::ALL {
        let idx = gem.index();
        if idx < 6 {
            payment.set(gem, info.max_replaceable[idx]);
        }
    }
    payment.set(GemType::Gold, info.mandatory_gold);

    Some(payment)
}

/// 根据自主分配的自由黄金方案计算实际扣款的标记集合
pub fn compute_custom_payment(
    player: &PlayerState,
    card: &JewelCard,
    allocated_gold: &[u8; 6],
) -> Option<TokenCollection> {
    let info = check_payment_divergence(player, card)?;
    let total_allocated: u8 = allocated_gold.iter().sum();
    if total_allocated > info.free_gold {
        return None;
    }

    let mut payment = TokenCollection::new();
    for gem in GemType::ALL {
        let idx = gem.index();
        if idx < 6 {
            let natural_pay = info.max_replaceable[idx];
            let gold_sub = allocated_gold[idx];
            if gold_sub > natural_pay {
                return None;
            }
            payment.set(gem, natural_pay - gold_sub);
        }
    }
    payment.set(GemType::Gold, info.mandatory_gold + total_allocated);

    Some(payment)
}
