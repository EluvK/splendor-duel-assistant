use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::heuristic_ai::HeuristicAI;
use super::neural_ai::NeuralAI;
use super::random_ai::RandomAI;
use super::replay::{
    action_category, format_action, DecisionDto, PlayerType, ReplaySession, ReplayStep,
    ScoredActionDto, StateDto,
};
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
        };
        sess.maybe_auto_skip_optional();
        sess
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
        *self = Self::new(seed, player_kinds);
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

    /// 若当前轮到 AI，执行一步 AI 演算并更新状态机
    pub fn step_ai(&mut self) -> Result<Option<ReplayStep>, String> {
        if matches!(self.game.phase, TurnPhase::GameOver(_)) {
            return Ok(None);
        }

        let player = self.game.current_player;
        let kind = self.player_kinds[player];

        let (action, decision) = match kind {
            PlayerKind::Human => return Ok(None),
            PlayerKind::Heuristic => {
                if let Some((best_act, score, scored_list)) =
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
                match NeuralAI::predict_action(&self.game, 1.0) {
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
                        // 回退到启发式 AI
                        if let Some((best_act, score, scored_list)) =
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
}
