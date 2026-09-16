use rand::prelude::*;

use crate::game_state::state::GameState;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;
use crate::model::card::CardAbility;
use crate::model::token::GemType;

/// 规则启发式智能体 (Heuristic AI)
pub struct HeuristicAI;

impl HeuristicAI {
    /// 评估所有合法动作并挑选综合得分最高者（带微小随机扰动避免死板对决）
    pub fn select_action<R: Rng + ?Sized>(state: &GameState, rng: &mut R) -> Option<Action> {
        Self::evaluate_and_select(state, rng).map(|(act, _, _)| act)
    }

    /// 评估所有合法动作，返回 (选中的最佳动作, 基础评分, 全部候选动作基础评分降序列表)
    pub fn evaluate_and_select<R: Rng + ?Sized>(
        state: &GameState,
        rng: &mut R,
    ) -> Option<(Action, f32, Vec<(Action, f32)>)> {
        let legals = RuleEngine::legal_actions(state);
        if legals.is_empty() {
            return None;
        }

        let mut scored_actions: Vec<(Action, f32)> = Vec::with_capacity(legals.len());
        let mut best_action = legals[0].clone();
        let mut best_noisy_score = f32::NEG_INFINITY;
        let mut best_base_score = f32::NEG_INFINITY;

        for act in legals {
            let base_score = Self::evaluate_action(state, &act);
            // 注入微弱噪声 [-0.5, 0.5] 打破平局
            let noise: f32 = rng.random_range(-0.5..0.5);
            let noisy_total = base_score + noise;

            if noisy_total > best_noisy_score {
                best_noisy_score = noisy_total;
                best_action = act.clone();
                best_base_score = base_score;
            }
            scored_actions.push((act, base_score));
        }

        // 按基础得分降序排序
        scored_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Some((best_action, best_base_score, scored_actions))
    }

    pub fn evaluate_action(state: &GameState, action: &Action) -> f32 {
        let cp = state.current_player;
        let p = &state.players[cp];

        match action {
            // -------------------------------------------------------------
            // 1. 购买卡牌 (核心得分行为)
            // -------------------------------------------------------------
            Action::PurchaseCard {
                from_reserved,
                tier,
                slot,
                plan_id,
            } => {
                let card = if *from_reserved {
                    if *slot < p.reserved_cards.len() {
                        p.reserved_cards[*slot].card
                    } else {
                        return -100.0;
                    }
                } else {
                    let t_idx = tier.index();
                    if *slot < state.pyramid[t_idx].len() {
                        state.pyramid[t_idx][*slot]
                    } else {
                        return -100.0;
                    }
                };

                // 斩杀判定：如果能直接带来胜利，赋予天量优先级
                if p.total_points + card.points >= 20 || p.total_crowns + card.crowns >= 10 {
                    return 10000.0;
                }
                if let Some(gem) = card.color.to_gem_type() {
                    if p.color_points[gem.index()] + card.points >= 10 {
                        return 10000.0;
                    }
                }

                let mut score = 80.0;
                // 高阶卡牌溢价 (鼓励升级发展与大卡斩杀)
                match tier {
                    crate::model::card::CardTier::Tier3 => score += 50.0,
                    crate::model::card::CardTier::Tier2 => score += 25.0,
                    crate::model::card::CardTier::Tier1 => {}
                }

                score += card.points as f32 * 40.0;
                score += card.crowns as f32 * 55.0; // 显著增强皇冠卡吸引力
                score += card.bonus as f32 * 15.0;

                // 皇冠冲刺激励：皇冠达到 3 顶或 6 顶王室门槛后，爆发式偏向皇冠卡
                if p.total_crowns >= 3 && card.crowns > 0 {
                    score += card.crowns as f32 * 45.0;
                }
                if p.total_crowns >= 6 && card.crowns > 0 {
                    score += card.crowns as f32 * 90.0;
                }

                // 单色冲刺激励：某单色达到 4 分以上时，全力集火同色高分卡
                if let Some(gem) = card.color.to_gem_type() {
                    let col_pts = p.color_points[gem.index()];
                    if col_pts >= 4 && card.points > 0 {
                        score += card.points as f32 * (30.0 + col_pts as f32 * 5.0);
                    }
                }

                // 逼近胜利时赋予更高紧迫度
                if p.total_points + card.points >= 16 || p.total_crowns + card.crowns >= 7 {
                    score += 80.0;
                }

                // 技能加权
                match card.ability {
                    Some(CardAbility::ExtraTurn) => score += 40.0,
                    Some(CardAbility::TakePrivilege) => score += 20.0,
                    Some(CardAbility::StealToken) => score += 20.0,
                    Some(CardAbility::TakeSameColor) => score += 15.0,
                    Some(CardAbility::ColorCopy) => score += 15.0,
                    Some(CardAbility::ColorCopyAndExtraTurn) => score += 55.0,
                    None => {}
                }

                // 预留卡购买稍加偏好（释放手牌槽位）
                if *from_reserved {
                    score += 5.0;
                }

                // 支付方案评分：默认方案微加分，花黄金保留珍珠加分
                if *plan_id == 0 {
                    score += 2.0;
                } else if (*plan_id as usize) < crate::gameplay::payment::PAYMENT_PLANS.len() {
                    let plan = &crate::gameplay::payment::PAYMENT_PLANS[*plan_id as usize];
                    if plan[GemType::Pearl.index()] > 0 {
                        score += 15.0;
                    }
                }

                score
            }

            // -------------------------------------------------------------
            // 2. 预留卡牌 (带拿黄金)
            // -------------------------------------------------------------
            Action::ReserveCard {
                gold_pos,
                tier,
                slot,
            } => {
                let dist_to_center = (gold_pos.0 as isize - 2).abs() + (gold_pos.1 as isize - 2).abs();
                let pos_bonus = 6.0 - dist_to_center as f32 * 1.5;
                let mut score = 40.0 + pos_bonus + 20.0;

                if let Some(s) = slot {
                    let t_idx = tier.index();
                    if *s < state.pyramid[t_idx].len() {
                        let c = state.pyramid[t_idx][*s];
                        let missing = p.tokens_missing(&c);

                        // 严防死锁：在缺乏 bonus 基础时，严禁无脑预留高费大牌将预留槽永久堵死！
                        if missing > 5 {
                            score -= (missing - 5) as f32 * 12.0;
                        }
                        if p.reserved_cards.len() >= 1 && missing > 3 {
                            score -= 20.0;
                        }
                        if p.reserved_cards.len() >= 2 && missing > 1 {
                            score -= 40.0;
                        }

                        score += c.points as f32 * 10.0;
                        score += c.crowns as f32 * 18.0;
                    }
                } else {
                    // 盲抽牌堆顶：卡牌不可预测且极易卡死手牌槽，风险极大！
                    // 降低基础评分，且已有预留牌时严惩盲抽，杜绝出现比明牌预留评分更高的倒挂现象
                    score -= 15.0;
                    if p.reserved_cards.len() >= 1 {
                        score -= 25.0;
                    }
                    if p.reserved_cards.len() >= 2 {
                        score -= 50.0;
                    }
                }
                score
            }

            // -------------------------------------------------------------
            // 3. 拿取标记 (资源积累)
            // -------------------------------------------------------------
            Action::TakeTokens { count, positions } => {
                let current_tokens = p.tokens.total();
                let total_after = current_tokens + *count;

                let mut score = 25.0 + (*count as f32 * 8.0);

                // 严惩溢出弃牌：超出 10 个的部分每个重罚 25 分
                if total_after > 10 {
                    let overflow = (total_after - 10) as f32;
                    score -= overflow * 25.0;
                }
                // 手牌达到 8~9 个时克制无脑盲目拿 3 连
                if current_tokens >= 8 && *count == 3 {
                    score -= 15.0;
                }

                let mut pearl_count = 0;
                let mut colors = Vec::with_capacity(3);

                for i in 0..*count as usize {
                    let (r, c) = positions[i];
                    if let Some(gem) = state.board.get(r, c) {
                        if gem == GemType::Pearl {
                            pearl_count += 1;
                            score += 15.0; // 珍珠稀缺
                        }
                        colors.push(gem);
                    }
                }

                // 目标协同加权：如果拿取的颜色是当前最接近能买到的卡牌所急需的，加分！
                // 栈上去重：避免同色 3 连重复遍历金字塔且避免多次重复叠加加权
                let mut unique_colors = [GemType::Gold; 3];
                let mut unique_len = 0;
                for &gem in &colors {
                    if !unique_colors[..unique_len].contains(&gem) {
                        unique_colors[unique_len] = gem;
                        unique_len += 1;
                    }
                }

                for &gem in &unique_colors[..unique_len] {
                    if Self::is_gem_needed_for_near_cards(state, p, gem, 3) {
                        score += 12.0;
                    }
                }

                // 惩罚特权扣分：给对手送特权需要适当惩罚
                let gives_privilege = (*count == 3 && colors[0] == colors[1] && colors[1] == colors[2])
                    || pearl_count >= 2;
                if gives_privilege {
                    score -= 20.0;
                }

                score
            }

            // -------------------------------------------------------------
            // 4. 特权卷轴与补板
            // -------------------------------------------------------------
            Action::UsePrivilege { r, c } => {
                let current_tokens = p.tokens.total();
                let gem = state.board.get(*r, *c);

                // 1. 手牌已达 10 枚：严禁使用特权，否则直接招致弃牌
                if current_tokens >= 10 {
                    return -50.0;
                }
                // 2. 手牌在 8~9 枚时：除非拿紧缺的珍珠，否则不滥用特权（0.0 < SkipOptional 15.0）
                if current_tokens >= 8 && gem != Some(GemType::Pearl) {
                    return 0.0;
                }

                if gem == Some(GemType::Pearl) {
                    40.0 // 拿珍珠极佳
                } else {
                    25.0
                }
            }

            Action::SkipOptional => 15.0,

            Action::ReplenishBoard => {
                let remaining = state.board.count_tokens();
                if remaining <= 4 {
                    45.0 // 盘面极度空竭，必须补板
                } else if remaining <= 7 {
                    20.0 // 适度补板
                } else {
                    -30.0 // 盘面仍有充足标记时补板纯属给对手白送特权
                }
            }

            // -------------------------------------------------------------
            // 5. 变色卡附着、偷标记与王室卡
            // -------------------------------------------------------------
            Action::AssignJokerColor { color } => {
                // 优先附着到当前已打出分数最高的基础颜色（争取单色 10 分胜利）
                let idx = color.index();
                50.0 + p.color_points[idx] as f32 * 10.0
            }

            Action::TakeSameColorToken { .. } => 50.0,

            Action::StealToken { gem } => {
                if *gem == GemType::Pearl {
                    60.0
                } else {
                    40.0
                }
            }

            Action::SelectRoyal { royal_id } => {
                match royal_id {
                    2 => 100.0, // 2分 + ExtraTurn
                    0 => 90.0,  // 2分 + Steal
                    1 => 85.0,  // 2分 + Privilege
                    _ => 80.0,  // 3分
                }
            }

            Action::DiscardToken { gem } => {
                // 黄金和珍珠极为珍贵，严防轻易丢弃；若极端情况下不得不弃，黄金优于珍珠保留
                match *gem {
                    GemType::Gold => return -90.0,
                    GemType::Pearl => return -100.0,
                    _ => {}
                }
                let mut score = p.tokens.get(*gem) as f32 * 5.0;
                // 保护急需颜色：若该宝石是场上最接近买到的卡牌所必需的，尽量不弃
                if Self::is_gem_needed_for_near_cards(state, p, *gem, 2) {
                    score -= 30.0;
                }
                score
            }
        }
    }

    /// 检查某种宝石是否属于当前最接近可购买卡牌（缺口 <= max_missing）的紧缺成本
    #[inline]
    fn is_gem_needed_for_near_cards(
        state: &GameState,
        p: &crate::game_state::player::PlayerState,
        gem: GemType,
        max_missing: u8,
    ) -> bool {
        state
            .pyramid
            .iter()
            .flat_map(|row| row.iter())
            .chain(p.reserved_cards.iter().map(|rc| &rc.card))
            .any(|c| {
                let missing = p.tokens_missing(c);
                if missing > 0 && missing <= max_missing {
                    let cost_gem = c.cost.get(gem);
                    let have = p.tokens.get(gem) + p.get_bonus(gem);
                    cost_gem > have
                } else {
                    false
                }
            })
    }
}
