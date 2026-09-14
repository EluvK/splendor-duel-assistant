"""Splendor Duel AI package."""

from splendor_ai._engine import PyGameState, generate_heuristic_samples
from splendor_ai.arena import Arena, ArenaResult
from splendor_ai.dataset import (
    CompactBatch,
    CompactDataset,
    FastTensorLoader,
    ReplayBuffer,
    ShardedBuffer,
)
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.mcts import (
    Agent,
    HeuristicAgent,
    MCTSAgent,
    PolicyNetAgent,
    RandomAgent,
    RustMCTSAgent,
)
from splendor_ai.net import SplendorNet
from splendor_ai.progress import Progress
from splendor_ai.selfplay import (
    generate_heuristic_compact_batch,
    generate_rust_neural_mcts_compact_batch,
    generate_selfplay_compact_batch,
)
from splendor_ai.trainer import Trainer, TrainerConfig

__version__ = "0.1.0"

__all__ = [
    "PyGameState",
    "generate_heuristic_samples",
    "SplendorDuelEnv",
    "SplendorNet",
    "CompactBatch",
    "CompactDataset",
    "FastTensorLoader",
    "ReplayBuffer",
    "ShardedBuffer",
    "Progress",
    "Trainer",
    "TrainerConfig",
    "Agent",
    "MCTSAgent",
    "PolicyNetAgent",
    "HeuristicAgent",
    "RandomAgent",
    "RustMCTSAgent",
    "Arena",
    "ArenaResult",
    "generate_heuristic_compact_batch",
    "generate_rust_neural_mcts_compact_batch",
    "generate_selfplay_compact_batch",
]
