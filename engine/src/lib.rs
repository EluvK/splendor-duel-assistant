pub mod ai;
pub mod bridge;
pub mod game_state;
pub mod gameplay;
pub mod model;

pub use ai::{
    action_category, format_action, CardDto, DecisionDto, HeuristicAI, InteractiveSession,
    LegalActionDto, NeuralAI, PlayerDto, PlayerKind, PlayerType, RandomAI, ReplaySession,
    ReplayStep, RoyalDto, RustMCTS, ScoredActionDto, StateDto, TractNeuralEvaluator,
};
pub use bridge::{action_mask, action_mask_from_legals, action_to_id, encode_state, ACTION_SIZE, CARD_FEAT_DIM, OBS_SIZE};
#[cfg(feature = "python")]
pub use bridge::PyGameState;
pub use game_state::{Board, GameState, PlayerState, TurnPhase, VictoryReason};
pub use gameplay::{check_victory, compute_card_payment, GameEngine, RuleEngine};
pub use model::{
    Action, ALL_JEWEL_CARDS, ALL_ROYAL_CARDS, CardAbility, CardColor, CardCost, CardTier, GemType,
    JewelCard, ReservedCard, RoyalCard, TokenCollection,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bridge::PyGameState>()?;
    m.add_function(wrap_pyfunction!(bridge::pymod::generate_heuristic_samples, m)?)?;
    m.add_function(wrap_pyfunction!(bridge::pymod::generate_neural_mcts_samples, m)?)?;
    m.add_function(wrap_pyfunction!(bridge::pymod::generate_neural_mcts_match_samples, m)?)?;
    m.add_function(wrap_pyfunction!(bridge::pymod::generate_gpu_batched_neural_mcts_samples, m)?)?;
    m.add_function(wrap_pyfunction!(bridge::pymod::evaluate_neural_match, m)?)?;
    Ok(())
}
