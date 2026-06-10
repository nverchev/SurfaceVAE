"""Data pipeline package."""

from src.data.dataset import COMAData, N_FACES, N_VERTICES, get_splits
from src.data.split import COMADatasetSplit, Partitions
from src.data.structures import Input, Target, Output
from src.data.operators import build_operator

__all__ = [
    "Input",
    "Target",
    "Output",
    "COMAData",
    "COMADatasetSplit",
    "Partitions",
    "get_splits",
    "N_FACES",
    "N_VERTICES",
    "build_operator",
]
