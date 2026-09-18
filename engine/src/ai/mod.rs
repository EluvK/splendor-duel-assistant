pub mod heuristic_ai;
pub mod interactive;
#[cfg(feature = "native")]
pub mod mcts;
#[cfg(feature = "native")]
pub mod neural_ai;
#[cfg(feature = "native")]
pub mod neural_evaluator;
pub mod random_ai;
pub mod replay;
#[cfg(feature = "native")]
pub mod sampling;

pub use heuristic_ai::HeuristicAI;
pub use interactive::{find_matching_action, InteractiveSession, LegalActionDto, PlayerKind};
pub use random_ai::RandomAI;
pub use replay::{
    CardDto, DecisionDto, PlayerDto, PlayerType, ReplaySession, ReplayStep, RoyalDto,
    ScoredActionDto, StateDto, action_category, format_action,
};

#[cfg(feature = "native")]
pub use mcts::RustMCTS;
#[cfg(feature = "native")]
pub use neural_ai::NeuralAI;
#[cfg(feature = "native")]
pub use neural_evaluator::TractNeuralEvaluator;
#[cfg(feature = "native")]
pub use sampling::{
    CompactBatchSamples, ParallelMatchResult, evaluate_neural_match_parallel,
    sample_heuristic_games_parallel, sample_neural_mcts_games_parallel,
    sample_neural_mcts_match_games_parallel,
};
