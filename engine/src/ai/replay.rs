use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::random_ai::RandomAI;
use crate::game_state::phase::TurnPhase;
use crate::game_state::player::PlayerState;
use crate::game_state::state::GameState;
use crate::gameplay::engine::GameEngine;
use crate::model::action::Action;
use crate::model::card::{CardAbility, CardColor, JewelCard, RoyalAbility, RoyalCard};
use crate::model::token::GemType;

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
        }
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
            royal_cards: p.royal_cards.iter().map(RoyalDto::from).collect(),
        }
    }
}

/// 完整盘面状态 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDto {
    pub turn_number: u32,
    pub current_player: usize,
    pub phase: String,
    pub board: [[Option<String>; 5]; 5],
    pub pyramid: [Vec<CardDto>; 3],
    pub decks_count: [usize; 3],
    pub bag_count: usize,
    pub privilege_pool: u8,
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
            royal_cards: s.royal_cards.iter().map(RoyalDto::from).collect(),
            players: [p0_dto, p1_dto],
            winner: winner_str,
        }
    }
}

/// 单步历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStep {
    pub step_index: usize,
    pub player: usize,
    pub action_desc: String,
    pub phase: String,
    pub state: StateDto,
}

/// 对局回放会话
pub struct ReplaySession {
    pub seed: u64,
    pub live_game: GameState,
    pub rng: ChaCha8Rng,
    pub history: Vec<ReplayStep>,
}

impl ReplaySession {
    pub fn new(seed: u64) -> Self {
        let game = GameState::new_game(seed);
        let initial_dto = StateDto::from(&game);

        let initial_step = ReplayStep {
            step_index: 0,
            player: game.current_player,
            action_desc: "Game Started".to_string(),
            phase: initial_dto.phase.clone(),
            state: initial_dto,
        };

        Self {
            seed,
            live_game: game,
            rng: ChaCha8Rng::seed_from_u64(seed),
            history: vec![initial_step],
        }
    }

    /// 重置对局
    pub fn reset(&mut self, seed: u64) {
        *self = Self::new(seed);
    }

    /// 执行一步随机/策略动作并记录快照
    pub fn step(&mut self) -> Result<bool, String> {
        if matches!(self.live_game.phase, TurnPhase::GameOver(_)) {
            return Ok(false); // 已结束
        }

        let player = self.live_game.current_player;
        let phase_desc = format!("{:?}", self.live_game.phase);

        if let Some(action) = RandomAI::select_action(&self.live_game, &mut self.rng) {
            let action_desc = format_action(&action);
            GameEngine::step(&mut self.live_game, &action)?;

            let state_dto = StateDto::from(&self.live_game);
            let next_step = ReplayStep {
                step_index: self.history.len(),
                player,
                action_desc,
                phase: phase_desc,
                state: state_dto,
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

fn format_action(action: &Action) -> String {
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
        Action::ReserveCard { tier, slot } => match slot {
            Some(s) => format!("Reserve Tier {tier:?} slot {s}"),
            None => format!("Reserve Tier {tier:?} from Deck"),
        },
        Action::PurchaseCard {
            from_reserved,
            tier,
            slot,
        } => {
            if *from_reserved {
                format!("Purchase Reserved card #{slot}")
            } else {
                format!("Purchase Tier {tier:?} slot {slot}")
            }
        }
        Action::AssignJokerColor { color } => format!("Joker attach to {color:?}"),
        Action::TakeSameColorToken { r, c } => format!("Take Same Color token at ({r}, {c})"),
        Action::StealToken { gem } => format!("Steal {gem:?} from Opponent"),
        Action::SelectRoyal { royal_id } => format!("Claim Royal Card #{royal_id}"),
        Action::DiscardToken { gem } => format!("Discard {gem:?}"),
    }
}
