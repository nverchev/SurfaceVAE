"""Configuration initialization."""

from src.config.specs import ExperimentConfig
from src.config.environment import PROJECT_NAME, VERSION, ConfigPath
from src.config.experiment import Experiment
from src.config.hydra import hydra_main, get_experiment_config
from src.config.options import ModelOperators

__all__ = [
    'PROJECT_NAME',
    'VERSION',
    'ConfigPath',
    'Experiment',
    'ExperimentConfig',
    'ModelOperators',
    'get_experiment_config',
    'hydra_main',
]
