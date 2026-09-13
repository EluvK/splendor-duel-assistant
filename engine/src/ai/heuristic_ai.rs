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
            } => {
                let card = if *from_reserved {
                    if *slot < p.reserved_cards.len() {
                        p.reserved_cards[*slot]
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

                let mut score = 100.0;
                score += card.points as f32 * 30.0;
                score += card.crowns as f32 * 25.0;
                score += card.bonus as f32 * 15.0;

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

                score
            }

            // -------------------------------------------------------------
            // 2. 预留卡牌 (抢黄金 / 卡对手 / 储备关键牌)
            // -------------------------------------------------------------
            Action::ReserveCard { tier, slot } => {
                let mut score = 35.0;
                if state.board.has_gold() {
                    score += 20.0; // 抢黄金收益极大
                }

                if let Some(s) = slot {
                    let t_idx = tier.index();
                    if *s < state.pyramid[t_idx].len() {
                        let c = state.pyramid[t_idx][*s];
                        score += c.points as f32 * 8.0;
                        score += c.crowns as f32 * 10.0;
                    }
                }
                score
            }

            // -------------------------------------------------------------
            // 3. 拿取标记 (资源积累)
            // -------------------------------------------------------------
            Action::TakeTokens { count, positions } => {
                let mut score = 30.0 + (*count as f32 * 10.0); // 优先拿 3 连

                let mut pearl_count = 0;
                let mut colors = Vec::with_capacity(3);

                for i in 0..*count as usize {
                    let (r, c) = positions[i];
                    if let Some(gem) = state.board.get(r, c) {
                        if gem == GemType::Pearl {
                            pearl_count += 1;
                            score += 12.0; // 珍珠稀缺
                        }
                        colors.push(gem);
                    }
                }

                // 惩罚特权扣分：给对手送特权需要适当惩罚
                let gives_privilege = (*count == 3 && colors[0] == colors[1] && colors[1] == colors[2])
                    || pearl_count >= 2;
                if gives_privilege {
                    score -= 15.0;
                }

                score
            }

            // -------------------------------------------------------------
            // 4. 特权卷轴与补板
            // -------------------------------------------------------------
            Action::UsePrivilege { r, c } => {
                let gem = state.board.get(*r, *c);
                if gem == Some(GemType::Pearl) {
                    40.0 // 拿珍珠极佳
                } else {
                    25.0
                }
            }

            Action::SkipOptional => 15.0,

            Action::ReplenishBoard => {
                let remaining = state.board.count_tokens();
                if remaining <= 7 {
                    35.0 // 盘面空竭，必须补板
                } else {
                    -10.0 // 盘面还很满时补板纯属给对手送特权
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
                // 弃牌时优先丢弃非珍珠、黄金，且手中库存最多的标记
                if *gem == GemType::Pearl || *gem == GemType::Gold {
                    -50.0
                } else {
                    p.tokens.get(*gem) as f32 * 5.0
                }
            }
        }
    }
}
