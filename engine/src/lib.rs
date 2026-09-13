pub mod ai;
pub mod bridge;
pub mod game_state;
pub mod gameplay;
pub mod model;

pub use ai::{
    CardDto, DecisionDto, HeuristicAI, PlayerDto, PlayerType, RandomAI, ReplaySession, ReplayStep,
    RoyalDto, ScoredActionDto, StateDto,
};
pub use bridge::{action_mask, action_to_id, encode_state, ACTION_SIZE, OBS_SIZE};
#[cfg(feature = "python")]
pub use bridge::PyGameState;
pub use game_state::{Board, GameState, PlayerState, TurnPhase, VictoryReason};
pub use gameplay::{check_victory, compute_card_payment, GameEngine, RuleEngine};
pub use model::{
    Action, ALL_JEWEL_CARDS, ALL_ROYAL_CARDS, CardAbility, CardColor, CardTier, GemType, JewelCard,
    RoyalCard, TokenCollection,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bridge::PyGameState>()?;
    Ok(())
}
