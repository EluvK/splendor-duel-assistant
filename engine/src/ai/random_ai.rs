use rand::prelude::*;

use crate::game_state::state::GameState;
use crate::gameplay::rules::RuleEngine;
use crate::model::action::Action;

/// 随机策略智能体
pub struct RandomAI;

impl RandomAI {
    /// 从当前合法动作中随机挑选一个
    pub fn select_action<R: Rng + ?Sized>(state: &GameState, rng: &mut R) -> Option<Action> {
        let actions = RuleEngine::legal_actions(state);
        if actions.is_empty() {
            None
        } else {
            actions.choose(rng).cloned()
        }
    }
}
