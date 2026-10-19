"""Train package containing losses, metrics, learning schemas and model wrappers."""

from src.train.objectives import get_reconstruction_metrics, get_vae_loss
from src.train.learning_schema import get_learning_schema
from src.train.models import VAEEMAModel, VAEModel, load_extract_vae_module
from src.train.hooks import (
    register_checkpointing,
    register_early_stopping,
    register_pruning,
)
from src.train.loaders import COMADataLoader

__all__ = [
    'COMADataLoader',
    'VAEEMAModel',
    'VAEModel',
    'get_learning_schema',
    'get_reconstruction_metrics',
    'get_vae_loss',
    'load_extract_vae_module',
    'register_checkpointing',
    'register_early_stopping',
    'register_pruning',
]
