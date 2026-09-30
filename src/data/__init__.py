"""Data pipeline package."""

from src.data.coma import (
    BaseCOMAData,
    COMAInterpolationData,
    COMAExtrapolationData,
    N_FACES,
    N_VERTICES,
)
from src.data.dataset import get_active_dataset, get_splits
from src.data.split import COMADatasetSplit, Partitions
from src.data.structures import Inputs, Targets, Outputs
from src.data.operators import (
    build_operator,
    get_dummy_operator,
    compute_operators_on_the_fly,
)

__all__ = [
    'N_FACES',
    'N_VERTICES',
    'BaseCOMAData',
    'COMADatasetSplit',
    'COMAExtrapolationData',
    'COMAInterpolationData',
    'Inputs',
    'Outputs',
    'Partitions',
    'Targets',
    'build_operator',
    'compute_operators_on_the_fly',
    'get_active_dataset',
    'get_dummy_operator',
    'get_splits',
]
