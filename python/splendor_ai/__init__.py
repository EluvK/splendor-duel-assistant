"""Splendor Duel AI package."""

from splendor_ai._engine import PyGameState
from splendor_ai.dataset import ReplayBuffer, Sample, SplendorDataset
from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet
from splendor_ai.selfplay import generate_heuristic_dataset, generate_selfplay_dataset
from splendor_ai.trainer import Trainer, TrainerConfig

__version__ = "0.1.0"

__all__ = [
    "PyGameState",
    "SplendorDuelEnv",
    "SplendorNet",
    "Sample",
    "SplendorDataset",
    "ReplayBuffer",
    "Trainer",
    "TrainerConfig",
    "generate_heuristic_dataset",
    "generate_selfplay_dataset",
]
