"""Data pipeline package."""

from src.data.coma import (
    BaseCOMAData,
    COMAData,
    COMAExtrapolationData,
    N_FACES,
    N_VERTICES,
)
from src.data.dataset import get_active_dataset, get_splits
from src.data.split import COMADatasetSplit, Partitions
from src.data.structures import Input, Target, Output
from src.data.operators import (
    build_operator,
    get_dummy_operator,
)

__all__ = [
    "Input",
    "Target",
    "Output",
    "BaseCOMAData",
    "COMAData",
    "COMAExtrapolationData",
    "get_active_dataset",
    "COMADatasetSplit",
    "Partitions",
    "get_splits",
    "N_FACES",
    "N_VERTICES",
    "build_operator",
    "get_dummy_operator",
]
