"""COMA Dataset loading and splitting logic."""

from typing import TYPE_CHECKING

from src.config import Experiment
from src.config.options import Datasets
from src.data.coma import BaseCOMAData, COMAData, COMAExtrapolationData
from src.data.split import Partitions

if TYPE_CHECKING:
    from src.data.split import COMADatasetSplit


def get_active_dataset() -> BaseCOMAData:
    """Get the active COMA dataset loader instance."""
    cfg = Experiment.get_config()
    if cfg.data.dataset.name == Datasets.COMA_EXTRAPOLATION:
        return COMAExtrapolationData()

    return COMAData()


def _get_splits() -> tuple["COMADatasetSplit", "COMADatasetSplit"]:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    dataset_loader = get_active_dataset()
    train_dataset = dataset_loader.get_split(train_partition)
    eval_dataset = dataset_loader.get_split(eval_partition)
    return train_dataset, eval_dataset


def get_splits() -> tuple["COMADatasetSplit", "COMADatasetSplit"]:
    """Get training and evaluation dataloaders in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    splits: tuple["COMADatasetSplit", "COMADatasetSplit"] | None = None
    if cfg.user.n_subprocesses:
        import torch.distributed as dist

        rank = dist.get_rank()
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                splits = _get_splits()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])
    else:
        splits = _get_splits()

    if splits is None:
        raise RuntimeError("Splits could not be created.")

    return splits
