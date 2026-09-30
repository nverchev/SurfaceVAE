"""Configuration initialization."""

from src.config.specs import AllConfig
from src.config.environment import PROJECT_NAME, VERSION, ConfigPath
from src.config.experiment import Experiment, get_trackers, set_tuning_logging
from src.config.hydra import hydra_main, get_config_all
from src.config.options import ModelOperators

__all__ = [
    'PROJECT_NAME',
    'VERSION',
    'AllConfig',
    'ConfigPath',
    'Experiment',
    'ModelOperators',
    'get_config_all',
    'get_trackers',
    'hydra_main',
    'set_tuning_logging',
]
