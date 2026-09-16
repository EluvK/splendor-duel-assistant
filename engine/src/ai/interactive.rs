use std::sync::Arc;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::heuristic_ai::HeuristicAI;
use super::mcts::{NeuralEvalCache, RustMCTS};
use super::neural_ai::NeuralAI;
use super::neural_evaluator::TractNeuralEvaluator;
use super::random_ai::RandomAI;
use super::replay::{
    action_category, format_action, DecisionDto, PlayerType, ReplaySession, ReplayStep,
    ScoredActionDto, StateDto,
};
use crate::bridge::{action_to_id, encode_state, ACTION_SIZE};
use crate::game_state::{GameState, TurnPhase};
use crate::gameplay::{GameEngine, RuleEngine};
use crate::model::Action;

/// 交互对战玩家身份类别
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayerKind {
    Human,
    Neural,
    Heuristic,
    Random,
}

impl PlayerKind {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "human" | "h" => Self::Human,
            "neural" | "nn" => Self::Neural,
            "random" | "rand" => Self::Random,
            _ => Self::Heuristic,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Neural => "neural",
            Self::Heuristic => "heuristic",
            Self::Random => "random",
        }
    }

    pub fn to_replay_player_type(&self) -> PlayerType {
        match self {
            Self::Human => PlayerType::Heuristic, // 回放时对于人类步数以启发式记录
            Self::Neural => PlayerType::Neural,
            Self::Heuristic => PlayerType::Heuristic,
            Self::Random => PlayerType::Random,
        }
    }
}

/// 面向前端的合法动作展示与交互传输结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalActionDto {
    pub action: Action,
    pub desc: String,
    pub category: String,
}

/// 交互对战会话
pub struct InteractiveSession {
    pub seed: u64,
    pub player_kinds: [PlayerKind; 2],
    pub game: GameState,
    pub rng: ChaCha8Rng,
    pub history: Vec<ReplayStep>,
    pub mcts_simulations: usize,
    pub evaluator: Option<Arc<TractNeuralEvaluator>>,
}

impl InteractiveSession {
    pub fn new(seed: u64, player_kinds: [PlayerKind; 2]) -> Self {
        let game = GameState::new_game(seed);
        let initial_dto = StateDto::from(&game);

        let initial_step = ReplayStep {
            step_index: 0,
            round_number: game.turn_number,
            player: game.current_player,
            action_desc: "Game Started".to_string(),
            phase: initial_dto.phase.clone(),
            state: initial_dto,
            decision: None,
        };

        let mut sess = Self {
            seed,
            player_kinds,
            game,
            rng: ChaCha8Rng::seed_from_u64(seed),
            history: vec![initial_step],
            mcts_simulations: 30,
            evaluator: None,
        };
        sess.maybe_auto_skip_optional();
        sess
    }

    /// 设置并挂载神经网络评估器 (供 Neural MCTS 深度推演使用)
    pub fn set_evaluator(&mut self, evaluator: Option<Arc<TractNeuralEvaluator>>) {
        self.evaluator = evaluator;
    }

    /// 设置默认 MCTS 模拟搜索强度
    pub fn set_mcts_simulations(&mut self, sims: usize) {
        self.mcts_simulations = sims;
    }

    /// 如果处于可选阶段且没有任何可执行的可选操作（无特权且不可补盘），自动跳过进入强制行动阶段
    pub fn maybe_auto_skip_optional(&mut self) {
        while self.game.phase == TurnPhase::OptionalActions {
            let legals = RuleEngine::legal_actions(&self.game);
            if legals.len() == 1 && legals[0] == Action::SkipOptional {
                let round_number = self.game.turn_number;
                let player = self.game.current_player;
                if GameEngine::step(&mut self.game, &Action::SkipOptional).is_ok() {
                    let next_step = ReplayStep {
                        step_index: self.history.len(),
                        round_number,
                        player,
                        action_desc: "Skip Optional (Auto)".to_string(),
                        phase: "OptionalActions".to_string(),
                        state: StateDto::from(&self.game),
                        decision: None,
                    };
                    self.history.push(next_step);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// 重置对局并保留或修改玩家配置
    pub fn reset(&mut self, seed: u64, player_kinds: [PlayerKind; 2]) {
        let evaluator = self.evaluator.clone();
        let mcts_simulations = self.mcts_simulations;
        *self = Self::new(seed, player_kinds);
        self.evaluator = evaluator;
        self.mcts_simulations = mcts_simulations;
    }

    /// 获取当前行动方身份类别
    pub fn current_player_kind(&self) -> PlayerKind {
        self.player_kinds[self.game.current_player]
    }

    /// 判断当前行动方是否为人类玩家
    pub fn is_current_player_human(&self) -> bool {
        self.current_player_kind() == PlayerKind::Human
    }

    /// 获取当前最新盘面 DTO
    pub fn current_state(&self) -> StateDto {
        StateDto::from(&self.game)
    }

    /// 获取所有合法动作
    pub fn legal_actions(&self) -> Vec<Action> {
        RuleEngine::legal_actions(&self.game)
    }

    /// 获取合法动作 DTO 列表
    pub fn legal_actions_dto(&self) -> Vec<LegalActionDto> {
        RuleEngine::legal_actions(&self.game)
            .into_iter()
            .map(|act| LegalActionDto {
                desc: format_action(&act),
                category: action_category(&act).to_string(),
                action: act,
            })
            .collect()
    }

    /// 人类玩家提交执行动作
    pub fn step_human(&mut self, action: Action) -> Result<ReplayStep, String> {
        if matches!(self.game.phase, TurnPhase::GameOver(_)) {
            return Err("Game already over".to_string());
        }

        let legals = self.legal_actions();
        if !legals.contains(&action) {
            return Err(format!("Illegal action: {:?}", action));
        }

        let player = self.game.current_player;
        let round_number = self.game.turn_number;
        let action_desc = format_action(&action);
        let phase_desc = format!("{:?}", self.game.phase);

        GameEngine::step(&mut self.game, &action)?;

        let next_step = ReplayStep {
            step_index: self.history.len(),
            round_number,
            player,
            action_desc,
            phase: phase_desc,
            state: StateDto::from(&self.game),
            decision: Some(DecisionDto {
                ai_type: "human".to_string(),
                chosen_score: None,
                top_candidates: vec![],
            }),
        };

        self.history.push(next_step.clone());
        self.maybe_auto_skip_optional();
        Ok(next_step)
    }

    /// 若当前轮到 AI，执行一步 AI 演算并更新状态机 (使用默认推演强度)
    pub fn step_ai(&mut self) -> Result<Option<ReplayStep>, String> {
        self.step_ai_with_sims(None)
    }

    /// 若当前轮到 AI，指定 MCTS 搜索强度执行一步 AI 演算并更新状态机
    pub fn step_ai_with_sims(&mut self, mcts_sims: Option<usize>) -> Result<Option<ReplayStep>, String> {
        if matches!(self.game.phase, TurnPhase::GameOver(_)) {
            return Ok(None);
        }

        let player = self.game.current_player;
        let kind = self.player_kinds[player];
        let sims = mcts_sims.unwrap_or(self.mcts_simulations);

        let (action, decision) = match kind {
            PlayerKind::Human => return Ok(None),
            PlayerKind::Heuristic => {
                if sims > 0 {
                    let mcts = RustMCTS::new(1.5, 15);
                    let legals = RuleEngine::legal_actions(&self.game);
                    if legals.is_empty() {
                        (None, None)
                    } else if legals.len() == 1 {
                        let chosen = legals[0].clone();
                        let decision = DecisionDto {
                            ai_type: format!("heuristic mcts ({} sims)", sims),
                            chosen_score: None,
                            top_candidates: vec![ScoredActionDto {
                                action_desc: format_action(&chosen),
                                score: 100.0,
                                is_chosen: true,
                            }],
                        };
                        (Some(chosen), Some(decision))
                    } else if let Some((best_act, policy)) = mcts.search_with_exploration_policy(
                        &self.game,
                        sims,
                        false,
                        0.3,
                        0.25,
                        0.0,
                        &mut self.rng,
                    ) {
                        let mut scored: Vec<(Action, f32)> = legals
                            .into_iter()
                            .map(|a| {
                                let id = action_to_id(&a);
                                let p = if id < ACTION_SIZE { policy[id] } else { 0.0 };
                                (a, p)
                            })
                            .collect();
                        scored.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        let top_candidates: Vec<ScoredActionDto> = scored
                            .into_iter()
                            .take(8)
                            .map(|(act, p)| ScoredActionDto {
                                action_desc: format_action(&act),
                                score: p * 100.0,
                                is_chosen: act == best_act,
                            })
                            .collect();

                        let decision = DecisionDto {
                            ai_type: format!("heuristic mcts ({} sims)", sims),
                            chosen_score: None,
                            top_candidates,
                        };
                        (Some(best_act), Some(decision))
                    } else if let Some((best_act, score, scored_list)) =
                        HeuristicAI::evaluate_and_select(&self.game, &mut self.rng)
                    {
                        let top_candidates: Vec<ScoredActionDto> = scored_list
                            .iter()
                            .take(8)
                            .map(|(act, s)| ScoredActionDto {
                                action_desc: format_action(act),
                                score: *s,
                                is_chosen: act == &best_act,
                            })
                            .collect();

                        let decision = DecisionDto {
                            ai_type: "heuristic (fallback)".to_string(),
                            chosen_score: Some(score),
                            top_candidates,
                        };
                        (Some(best_act), Some(decision))
                    } else {
                        (None, None)
                    }
                } else if let Some((best_act, score, scored_list)) =
                    HeuristicAI::evaluate_and_select(&self.game, &mut self.rng)
                {
                    let top_candidates: Vec<ScoredActionDto> = scored_list
                        .iter()
                        .take(8)
                        .map(|(act, s)| ScoredActionDto {
                            action_desc: format_action(act),
                            score: *s,
                            is_chosen: act == &best_act,
                        })
                        .collect();

                    let decision = DecisionDto {
                        ai_type: "heuristic".to_string(),
                        chosen_score: Some(score),
                        top_candidates,
                    };
                    (Some(best_act), Some(decision))
                } else {
                    (None, None)
                }
            }
            PlayerKind::Random => {
                let act = RandomAI::select_action(&self.game, &mut self.rng);
                let decision = DecisionDto {
                    ai_type: "random".to_string(),
                    chosen_score: None,
                    top_candidates: vec![],
                };
                (act, Some(decision))
            }
            PlayerKind::Neural => {
                if sims > 0 && self.evaluator.is_some() {
                    let evaluator = self.evaluator.as_ref().unwrap();
                    let legals = RuleEngine::legal_actions(&self.game);
                    if legals.is_empty() {
                        (None, None)
                    } else if legals.len() == 1 {
                        let chosen = legals[0].clone();
                        let decision = DecisionDto {
                            ai_type: format!("neural mcts ({} sims)", sims),
                            chosen_score: None,
                            top_candidates: vec![ScoredActionDto {
                                action_desc: format_action(&chosen),
                                score: 100.0,
                                is_chosen: true,
                            }],
                        };
                        (Some(chosen), Some(decision))
                    } else {
                        let mcts = RustMCTS::new(1.5, 15);
                        let mut eval_cache = NeuralEvalCache::default();
                        let search_res = mcts.search_neural_policy_with_legals_and_cache(
                            &self.game,
                            legals.clone(),
                            evaluator,
                            &mut eval_cache,
                            sims,
                            false,
                            0.3,
                            0.25,
                            0.0,
                            &mut self.rng,
                        );

                        if let Some((best_act, policy)) = search_res {
                            let mut scored: Vec<(Action, f32)> = legals
                                .into_iter()
                                .map(|a| {
                                    let id = action_to_id(&a);
                                    let p = if id < ACTION_SIZE { policy[id] } else { 0.0 };
                                    (a, p)
                                })
                                .collect();
                            scored.sort_unstable_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });

                            let top_candidates: Vec<ScoredActionDto> = scored
                                .into_iter()
                                .take(8)
                                .map(|(act, p)| ScoredActionDto {
                                    action_desc: format_action(&act),
                                    score: p * 100.0,
                                    is_chosen: act == best_act,
                                })
                                .collect();

                            let root_obs = encode_state(&self.game);
                            let root_val = evaluator
                                .evaluate(&root_obs)
                                .ok()
                                .map(|p| p.1.win_value * 100.0);

                            let decision = DecisionDto {
                                ai_type: format!("neural mcts ({} sims)", sims),
                                chosen_score: root_val,
                                top_candidates,
                            };
                            (Some(best_act), Some(decision))
                        } else {
                            Self::predict_single_step_neural(&self.game, &mut self.rng)
                        }
                    }
                } else {
                    Self::predict_single_step_neural(&self.game, &mut self.rng)
                }
            }
        };

        if let Some(action) = action {
            let round_number = self.game.turn_number;
            let action_desc = format_action(&action);
            let phase_desc = format!("{:?}", self.game.phase);

            GameEngine::step(&mut self.game, &action)?;

            let next_step = ReplayStep {
                step_index: self.history.len(),
                round_number,
                player,
                action_desc,
                phase: phase_desc,
                state: StateDto::from(&self.game),
                decision,
            };

            self.history.push(next_step.clone());
            self.maybe_auto_skip_optional();
            Ok(Some(next_step))
        } else {
            Err("No legal action for AI".to_string())
        }
    }

    /// 转换为可复盘的 ReplaySession 实例
    pub fn to_replay_session(&self) -> ReplaySession {
        let player_types = [
            self.player_kinds[0].to_replay_player_type(),
            self.player_kinds[1].to_replay_player_type(),
        ];
        ReplaySession {
            seed: self.seed,
            player_types,
            live_game: self.game.clone(),
            rng: self.rng.clone(),
            history: self.history.clone(),
        }
    }

    /// 单步纯直觉神经网络推理辅助函数 (0 次 MCTS 推演或回退时使用)
    fn predict_single_step_neural(
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> (Option<Action>, Option<DecisionDto>) {
        match NeuralAI::predict_action(game, 1.0) {
            Ok(pred) => {
                let top_candidates: Vec<ScoredActionDto> = pred
                    .top_candidates
                    .into_iter()
                    .map(|(act, prob, is_chosen)| ScoredActionDto {
                        action_desc: format_action(&act),
                        score: prob * 100.0,
                        is_chosen,
                    })
                    .collect();

                let decision = DecisionDto {
                    ai_type: format!("neural (epoch {})", pred.epoch),
                    chosen_score: Some(pred.winrate * 100.0),
                    top_candidates,
                };
                (Some(pred.best_action), Some(decision))
            }
            Err(err_msg) => {
                if let Some((best_act, score, scored_list)) =
                    HeuristicAI::evaluate_and_select(game, rng)
                {
                    let top_candidates: Vec<ScoredActionDto> = scored_list
                        .iter()
                        .take(8)
                        .map(|(act, s)| ScoredActionDto {
                            action_desc: format_action(act),
                            score: *s,
                            is_chosen: act == &best_act,
                        })
                        .collect();

                    let decision = DecisionDto {
                        ai_type: format!("neural (fallback: {err_msg})"),
                        chosen_score: Some(score),
                        top_candidates,
                    };
                    (Some(best_act), Some(decision))
                } else {
                    (None, None)
                }
            }
        }
    }
}
