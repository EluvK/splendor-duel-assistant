use crate::game_state::player::PlayerState;
use crate::model::card::JewelCard;
use crate::model::token::{GemType, TokenCollection};

/// 支付分歧与需求分析信息
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

/// 编译期生成全量 84 种自由黄金分配方案 (6 种资源分配最多 3 枚自由黄金的所有多重组合)
pub const fn generate_payment_plans() -> [[u8; 6]; 84] {
    let mut plans = [[0u8; 6]; 84];
    let mut idx = 0;

    // k = 0 (1 种): 全部为 0
    idx += 1;

    // k = 1 (6 种): 6 种颜色各取 1 枚
    let mut c0 = 0;
    while c0 < 6 {
        plans[idx][c0] += 1;
        idx += 1;
        c0 += 1;
    }

    // k = 2 (21 种): 6 种颜色选 2 枚 (带放回)
    let mut c0 = 0;
    while c0 < 6 {
        let mut c1 = c0;
        while c1 < 6 {
            plans[idx][c0] += 1;
            plans[idx][c1] += 1;
            idx += 1;
            c1 += 1;
        }
        c0 += 1;
    }

    // k = 3 (56 种): 6 种颜色选 3 枚 (带放回)
    let mut c0 = 0;
    while c0 < 6 {
        let mut c1 = c0;
        while c1 < 6 {
            let mut c2 = c1;
            while c2 < 6 {
                plans[idx][c0] += 1;
                plans[idx][c1] += 1;
                plans[idx][c2] += 1;
                idx += 1;
                c2 += 1;
            }
            c1 += 1;
        }
        c0 += 1;
    }

    assert!(idx == 84);
    plans
}

/// 静态全量 84 种自由黄金替代方案 (Plan 0 为默认方案 [0, 0, 0, 0, 0, 0])
pub const PAYMENT_PLANS: [[u8; 6]; 84] = generate_payment_plans();

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

/// 判定特定支付方案是否对该卡牌和玩家合法
pub fn is_legal_payment_plan(player: &PlayerState, card: &JewelCard, plan_id: u8) -> bool {
    if plan_id >= 84 {
        return false;
    }
    let info = match check_payment_divergence(player, card) {
        Some(info) => info,
        None => return false,
    };

    // Plan 0 为默认方案，只要买得起即合法
    if plan_id == 0 {
        return true;
    }

    let plan = &PAYMENT_PLANS[plan_id as usize];
    let total_sub: u8 = plan[0] + plan[1] + plan[2] + plan[3] + plan[4] + plan[5];
    if total_sub > info.free_gold {
        return false;
    }

    for i in 0..6 {
        if plan[i] > info.max_replaceable[i] {
            return false;
        }
    }

    true
}

/// 计算当前卡牌对当前玩家的所有合法支付方案的 u128 位掩码 (0..84 位，极速零堆分配)
pub fn legal_payment_plans_mask(player: &PlayerState, card: &JewelCard) -> u128 {
    let info = match check_payment_divergence(player, card) {
        Some(info) => info,
        None => return 0,
    };

    let mut mask = 1u128; // Plan 0 必定合法

    // 若无自由黄金或无可替代天然宝石，则仅 Plan 0 合法
    if !info.has_divergence {
        return mask;
    }

    for (plan_id, plan) in PAYMENT_PLANS.iter().enumerate().skip(1) {
        let total_sub: u8 = plan[0] + plan[1] + plan[2] + plan[3] + plan[4] + plan[5];
        if total_sub > info.free_gold {
            continue;
        }

        let mut valid = true;
        for i in 0..6 {
            if plan[i] > info.max_replaceable[i] {
                valid = false;
                break;
            }
        }
        if valid {
            mask |= 1u128 << plan_id;
        }
    }

    mask
}

/// 获取当前卡牌对当前玩家的所有合法支付方案 ID 列表
pub fn legal_payment_plans(player: &PlayerState, card: &JewelCard) -> Vec<u8> {
    let mask = legal_payment_plans_mask(player, card);
    if mask == 0 {
        return Vec::new();
    }
    let mut plans = Vec::with_capacity(16);
    for p in 0..84 {
        if (mask & (1u128 << p)) != 0 {
            plans.push(p as u8);
        }
    }
    plans
}

/// 根据指定的支付方案计算实际扣款的标记集合
pub fn compute_payment_with_plan(
    player: &PlayerState,
    card: &JewelCard,
    plan_id: u8,
) -> Option<TokenCollection> {
    if plan_id >= 84 {
        return None;
    }
    let info = check_payment_divergence(player, card)?;
    let plan = &PAYMENT_PLANS[plan_id as usize];
    let total_sub: u8 = plan[0] + plan[1] + plan[2] + plan[3] + plan[4] + plan[5];
    if total_sub > info.free_gold {
        return None;
    }

    let mut payment = TokenCollection::new();
    for gem in GemType::ALL {
        let idx = gem.index();
        if idx < 6 {
            let natural_pay = info.max_replaceable[idx];
            let gold_sub = plan[idx];
            if gold_sub > natural_pay {
                return None;
            }
            payment.set(gem, natural_pay - gold_sub);
        }
    }
    payment.set(GemType::Gold, info.mandatory_gold + total_sub);

    Some(payment)
}

/// 计算购买卡牌所需支付的标记集合（默认最优方案：Plan 0）
pub fn compute_card_payment(player: &PlayerState, card: &JewelCard) -> Option<TokenCollection> {
    compute_payment_with_plan(player, card, 0)
}
