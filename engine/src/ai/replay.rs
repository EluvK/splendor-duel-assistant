use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::heuristic_ai::HeuristicAI;
#[cfg(feature = "native")]
use super::neural_ai::NeuralAI;
use super::random_ai::RandomAI;
use crate::game_state::phase::TurnPhase;
use crate::game_state::player::PlayerState;
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::model::action::Action;
use crate::model::card::{
    CardAbility, CardColor, JewelCard, ReservedCard, RoyalAbility, RoyalCard,
};
use crate::model::token::GemType;

fn default_is_public() -> bool {
    true
}

/// 供前端渲染的卡牌 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardDto {
    pub id: u8,
    pub tier: usize, // 1, 2, 3
    pub color: String,
    pub points: u8,
    pub bonus: u8,
    pub crowns: u8,
    pub ability: Option<String>,
    pub cost: CardCostDto,
    #[serde(default = "default_is_public")]
    pub is_public: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardCostDto {
    pub white: u8,
    pub blue: u8,
    pub green: u8,
    pub red: u8,
    pub black: u8,
    pub pearl: u8,
}

impl From<&JewelCard> for CardDto {
    fn from(c: &JewelCard) -> Self {
        let color_str = match c.color {
            CardColor::White => "white",
            CardColor::Blue => "blue",
            CardColor::Green => "green",
            CardColor::Red => "red",
            CardColor::Black => "black",
            CardColor::Points => "points",
            CardColor::Joker => "joker",
        };

        let ab_str = c.ability.map(|ab| match ab {
            CardAbility::ExtraTurn => "ExtraTurn".to_string(),
            CardAbility::TakePrivilege => "TakePrivilege".to_string(),
            CardAbility::TakeSameColor => "TakeSameColor".to_string(),
            CardAbility::StealToken => "StealToken".to_string(),
            CardAbility::ColorCopy => "ColorCopy".to_string(),
            CardAbility::ColorCopyAndExtraTurn => "ColorCopyAndExtraTurn".to_string(),
        });

        Self {
            id: c.id,
            tier: c.tier.index() + 1,
            color: color_str.to_string(),
            points: c.points,
            bonus: c.bonus,
            crowns: c.crowns,
            ability: ab_str,
            cost: CardCostDto {
                white: c.cost.white,
                blue: c.cost.blue,
                green: c.cost.green,
                red: c.cost.red,
                black: c.cost.black,
                pearl: c.cost.pearl,
            },
            is_public: true,
        }
    }
}

impl From<&ReservedCard> for CardDto {
    fn from(rc: &ReservedCard) -> Self {
        let mut dto = CardDto::from(&rc.card);
        dto.is_public = rc.is_public;
        dto
    }
}

/// 王室卡 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoyalDto {
    pub id: u8,
    pub points: u8,
    pub ability: Option<String>,
}

impl From<&RoyalCard> for RoyalDto {
    fn from(r: &RoyalCard) -> Self {
        let ab_str = r.ability.map(|ab| match ab {
            RoyalAbility::StealToken => "StealToken".to_string(),
            RoyalAbility::TakePrivilege => "TakePrivilege".to_string(),
            RoyalAbility::ExtraTurn => "ExtraTurn".to_string(),
        });
        Self {
            id: r.id,
            points: r.points,
            ability: ab_str,
        }
    }
}

/// 玩家状态 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerDto {
    pub id: usize,
    pub tokens: [u8; 7], // White, Blue, Green, Red, Black, Pearl, Gold
    pub token_total: u8,
    pub bonuses: [u8; 5],
    pub color_points: [u8; 5],
    pub total_points: u8,
    pub total_crowns: u8,
    pub privileges: u8,
    pub reserved_cards: Vec<CardDto>,
    pub cards_count: usize,
    pub purchased_cards: Vec<CardDto>,
    pub royal_cards: Vec<RoyalDto>,
}

impl From<&PlayerState> for PlayerDto {
    fn from(p: &PlayerState) -> Self {
        Self {
            id: p.id,
            tokens: p.tokens.counts,
            token_total: p.tokens.total(),
            bonuses: p.bonuses,
            color_points: p.color_points,
            total_points: p.total_points,
            total_crowns: p.total_crowns,
            privileges: p.privileges,
            reserved_cards: p.reserved_cards.iter().map(CardDto::from).collect(),
            cards_count: p.cards.len(),
            purchased_cards: p.cards.iter().map(CardDto::from).collect(),
            royal_cards: p.royal_cards.iter().map(RoyalDto::from).collect(),
        }
    }
}

/// 完整盘面状态 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDto {
    pub turn_number: u32,
    pub round_number: u32,
    pub current_player: usize,
    pub phase: String,
    pub board: [[Option<String>; 5]; 5],
    pub pyramid: [Vec<CardDto>; 3],
    pub decks_count: [usize; 3],
    pub bag_count: usize,
    pub privilege_pool: u8,
    pub replenished_this_turn: bool,
    pub royal_cards: Vec<RoyalDto>,
    pub players: [PlayerDto; 2],
    pub winner: Option<String>,
}

impl From<&GameState> for StateDto {
    fn from(s: &GameState) -> Self {
        let mut board_dto: [[Option<String>; 5]; 5] = Default::default();
        for r in 0..5 {
            for c in 0..5 {
                board_dto[r][c] = s.board.get(r, c).map(|g| match g {
                    GemType::White => "white".to_string(),
                    GemType::Blue => "blue".to_string(),
                    GemType::Green => "green".to_string(),
                    GemType::Red => "red".to_string(),
                    GemType::Black => "black".to_string(),
                    GemType::Pearl => "pearl".to_string(),
                    GemType::Gold => "gold".to_string(),
                });
            }
        }

        let p0_dto = PlayerDto::from(&s.players[0]);
        let p1_dto = PlayerDto::from(&s.players[1]);

        let phase_str = match &s.phase {
            TurnPhase::OptionalActions => "OptionalActions".to_string(),
            TurnPhase::MandatoryAction => "MandatoryAction".to_string(),
            TurnPhase::CardAbilityJoker { .. } => "CardAbilityJoker".to_string(),
            TurnPhase::CardAbilitySameColor { color } => {
                format!("CardAbilitySameColor({color:?})")
            }
            TurnPhase::CardAbilitySteal => "CardAbilitySteal".to_string(),
            TurnPhase::SelectRoyalCard => "SelectRoyalCard".to_string(),
            TurnPhase::DiscardTokens => "DiscardTokens".to_string(),
            TurnPhase::GameOver(reason) => format!("GameOver({reason:?})"),
        };

        let winner_str = s
            .winner
            .map(|(p, reason)| format!("Player {p} ({reason:?})"));

        Self {
            turn_number: s.turn_number,
            round_number: s.turn_number,
            current_player: s.current_player,
            phase: phase_str,
            board: board_dto,
            pyramid: [
                s.pyramid[0].iter().map(CardDto::from).collect(),
                s.pyramid[1].iter().map(CardDto::from).collect(),
                s.pyramid[2].iter().map(CardDto::from).collect(),
            ],
            decks_count: [
                s.decks[0].len(),
                s.decks[1].len(),
                s.decks[2].len(),
            ],
            bag_count: s.bag.len(),
            privilege_pool: s.privilege_pool,
            replenished_this_turn: s.replenished_this_turn,
            royal_cards: s.royal_cards.iter().map(RoyalDto::from).collect(),
            players: [p0_dto, p1_dto],
            winner: winner_str,
        }
    }
}

/// 玩家 AI 类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayerType {
    Heuristic,
    Random,
    Neural, // 预留未来通过 Python 桥接的模型推断
}

impl PlayerType {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "random" => PlayerType::Random,
            "neural" => PlayerType::Neural,
            _ => PlayerType::Heuristic,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PlayerType::Heuristic => "heuristic",
            PlayerType::Random => "random",
            PlayerType::Neural => "neural",
        }
    }
}

/// 启发式打分项 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredActionDto {
    pub action_desc: String,
    pub score: f32,
    pub is_chosen: bool,
}

/// 单步 AI 思考与评分决策详情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDto {
    pub ai_type: String,
    pub chosen_score: Option<f32>,
    pub top_candidates: Vec<ScoredActionDto>,
}

/// 单步历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStep {
    pub step_index: usize,
    pub round_number: u32,
    pub player: usize,
    pub action_desc: String,
    pub phase: String,
    pub state: StateDto,
    pub decision: Option<DecisionDto>,
}

/// 对局回放会话
pub struct ReplaySession {
    pub seed: u64,
    pub player_types: [PlayerType; 2],
    pub live_game: GameState,
    pub rng: ChaCha8Rng,
    pub history: Vec<ReplayStep>,
}

impl ReplaySession {
    pub fn new(seed: u64) -> Self {
        Self::new_with_players(seed, [PlayerType::Heuristic, PlayerType::Heuristic])
    }

    pub fn new_with_players(seed: u64, player_types: [PlayerType; 2]) -> Self {
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

        Self {
            seed,
            player_types,
            live_game: game,
            rng: ChaCha8Rng::seed_from_u64(seed),
            history: vec![initial_step],
        }
    }

    /// 重置对局（保持当前玩家类型配置）
    pub fn reset(&mut self, seed: u64) {
        let types = self.player_types;
        *self = Self::new_with_players(seed, types);
    }

    /// 使用指定玩家类型配置重置对局
    pub fn reset_with_players(&mut self, seed: u64, player_types: [PlayerType; 2]) {
        *self = Self::new_with_players(seed, player_types);
    }

    /// 动态修改玩家 AI 类型
    pub fn set_player_type(&mut self, player_idx: usize, p_type: PlayerType) {
        if player_idx < 2 {
            self.player_types[player_idx] = p_type;
        }
    }

    /// 批量推进 N 步，返回实际推进的步数
    pub fn step_n(&mut self, n: usize) -> Result<usize, String> {
        let mut count = 0;
        for _ in 0..n {
            if matches!(self.live_game.phase, TurnPhase::GameOver(_)) {
                break;
            }
            if self.step()? {
                count += 1;
            } else {
                break;
            }
        }
        Ok(count)
    }

    /// 自动推进直到游戏结束或达到最大步数限制（如 1500 步）
    pub fn play_to_end(&mut self, max_steps: usize) -> Result<usize, String> {
        self.step_n(max_steps)
    }

    /// 执行一步动作并记录快照与启发式评分决策
    pub fn step(&mut self) -> Result<bool, String> {
        if matches!(self.live_game.phase, TurnPhase::GameOver(_)) {
            return Ok(false); // 已结束
        }

        let player = self.live_game.current_player;
        let ai_type = self.player_types[player];
        let phase_desc = format!("{:?}", self.live_game.phase);

        let (action, decision) = match ai_type {
            PlayerType::Heuristic => {
                if let Some((best_act, score, scored_list)) =
                    HeuristicAI::evaluate_and_select(&self.live_game, &mut self.rng)
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
            PlayerType::Random => {
                let act = RandomAI::select_action(&self.live_game, &mut self.rng);
                let decision = DecisionDto {
                    ai_type: "random".to_string(),
                    chosen_score: None,
                    top_candidates: vec![],
                };
                (act, Some(decision))
            }
            PlayerType::Neural => {
                #[cfg(feature = "native")]
                {
                    match NeuralAI::predict_action(&self.live_game, 1.0) {
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
                            // 推理服务未就绪时，优雅回退到启发式 AI 避免卡死
                            if let Some((best_act, score, scored_list)) =
                                HeuristicAI::evaluate_and_select(&self.live_game, &mut self.rng)
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
                #[cfg(not(feature = "native"))]
                {
                    if let Some((best_act, score, scored_list)) =
                        HeuristicAI::evaluate_and_select(&self.live_game, &mut self.rng)
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
                            ai_type: "neural (fallback: heuristic)".to_string(),
                            chosen_score: Some(score),
                            top_candidates,
                        };
                        (Some(best_act), Some(decision))
                    } else {
                        (None, None)
                    }
                }
            }
        };

        if let Some(action) = action {
            let round_number = self.live_game.turn_number;
            let action_desc = format_action(&action);
            GameEngine::step(&mut self.live_game, &action)?;

            let state_dto = StateDto::from(&self.live_game);
            let next_step = ReplayStep {
                step_index: self.history.len(),
                round_number,
                player,
                action_desc,
                phase: phase_desc,
                state: state_dto,
                decision,
            };
            self.history.push(next_step);
            Ok(true)
        } else {
            Err("No legal action available".to_string())
        }
    }

    /// 获取当前最新状态
    pub fn current_state(&self) -> StateDto {
        StateDto::from(&self.live_game)
    }

    /// 获取指定步数的历史快照
    pub fn get_step(&self, index: usize) -> Option<&ReplayStep> {
        self.history.get(index)
    }
}

pub fn action_category(action: &Action) -> &'static str {
    match action {
        Action::SkipOptional => "skip_optional",
        Action::UsePrivilege { .. } => "use_privilege",
        Action::ReplenishBoard => "replenish",
        Action::TakeTokens { .. } => "take_tokens",
        Action::ReserveCard { .. } => "reserve_card",
        Action::PurchaseCard { .. } => "purchase_card",
        Action::AssignJokerColor { .. } => "joker",
        Action::TakeSameColorToken { .. } => "same_color",
        Action::StealToken { .. } => "steal",
        Action::SelectRoyal { .. } => "royal",
        Action::DiscardToken { .. } => "discard",
    }
}

pub fn format_action(action: &Action) -> String {
    match action {
        Action::SkipOptional => "Skip Optional".to_string(),
        Action::UsePrivilege { r, c } => format!("Use Privilege ({r}, {c})"),
        Action::ReplenishBoard => "Replenish Board".to_string(),
        Action::TakeTokens { count, positions } => {
            let pts: Vec<_> = positions[0..*count as usize]
                .iter()
                .map(|(r, c)| format!("({r},{c})"))
                .collect();
            format!("Take {} Tokens: {}", count, pts.join(", "))
        }
        Action::ReserveCard { gold_pos, tier, slot } => {
            let (gr, gc) = gold_pos;
            match slot {
                Some(s) => format!("Reserve Tier {tier:?} slot {s} (gold at ({gr},{gc}))"),
                None => format!("Reserve Tier {tier:?} from Deck (gold at ({gr},{gc}))"),
            }
        }
        Action::PurchaseCard {
            from_reserved,
            tier,
            slot,
            plan_id,
        } => {
            let plan_str = if *plan_id == 0 {
                "default pay".to_string()
            } else {
                format!("plan #{plan_id}")
            };
            if *from_reserved {
                format!("Purchase Reserved card #{slot} [{plan_str}]")
            } else {
                format!("Purchase Tier {tier:?} slot {slot} [{plan_str}]")
            }
        }
        Action::AssignJokerColor { color } => format!("Joker attach to {color:?}"),
        Action::TakeSameColorToken { r, c } => format!("Take Same Color token at ({r}, {c})"),
        Action::StealToken { gem } => format!("Steal {gem:?} from Opponent"),
        Action::SelectRoyal { royal_id } => format!("Claim Royal Card #{royal_id}"),
        Action::DiscardToken { gem } => format!("Discard {gem:?}"),
    }
}
