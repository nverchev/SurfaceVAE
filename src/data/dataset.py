"""COMA Dataset loading and splitting logic."""

from src.config import Experiment
from src.config.options import Datasets
from src.data.coma import (
    DataBuilder,
    COMAInterpolationDataBuilder,
    COMAExtrapolationDataBuilder,
    COMAIdentityDataBuilder,
)
from src.data.split import Partitions, COMASplit


def get_databuilder() -> DataBuilder:
    """Get the active COMA dataset loader instance."""
    cfg = Experiment.get_config()
    if cfg.data.dataset.name == Datasets.COMA_EXTRAPOLATION:
        return COMAExtrapolationDataBuilder()

    if cfg.data.dataset.name == Datasets.COMA_IDENTITY:
        return COMAIdentityDataBuilder()

    return COMAInterpolationDataBuilder()


def _get_splits() -> tuple[COMASplit, COMASplit]:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    databuilder = get_databuilder()
    train_split = databuilder.get_split(train_partition)
    eval_split = databuilder.get_split(eval_partition)
    return train_split, eval_split


def get_splits() -> tuple[COMASplit, COMASplit]:
    """Get training and evaluation dataloaders in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    splits: tuple[COMASplit, COMASplit] | None = None
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
        raise RuntimeError('Splits could not be created.')

    return splits
