"""Data pipeline package."""

from src.data.coma import (
    DataBuilder,
    COMAInterpolationDataBuilder,
    COMAExtrapolationDataBuilder,
    N_FACES,
    N_VERTICES,
)
from src.data.dataset import get_databuilder, get_splits
from src.data.split import COMASplit, Partitions
from src.data.structures import Inputs, Targets, Outputs
from src.data.operators import (
    build_operator,
    get_dummy_operator,
    compute_operators_on_the_fly,
)

__all__ = [
    'N_FACES',
    'N_VERTICES',
    'COMAExtrapolationDataBuilder',
    'COMAInterpolationDataBuilder',
    'COMASplit',
    'DataBuilder',
    'Inputs',
    'Outputs',
    'Partitions',
    'Targets',
    'build_operator',
    'compute_operators_on_the_fly',
    'get_databuilder',
    'get_dummy_operator',
    'get_splits',
]
