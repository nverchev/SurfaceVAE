"""Main script for running the PCA experiment on the dataset."""

import logging
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import PCA

from drytorch.core.log_events import MetricEvent

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.data import get_splits

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


DEBUG_MODE = (
    any(module in sys.modules for module in ("pydevd", "debugpy", "pdb"))
    or sys.gettrace() is not None
)

logger = logging.getLogger(__name__)


def train_pca() -> None:
    """Fit PCA on training split and evaluate reconstruction error."""
    cfg = Experiment.get_config()
    train_ds, eval_ds = get_splits()
    train_shapes = train_ds._shapes
    eval_shapes = eval_ds._shapes
    mean = train_shapes.mean(axis=0)
    std = train_shapes.std(axis=0)
    train_normalized = (train_shapes - mean) / (std + 1e-8)
    eval_normalized = (eval_shapes - mean) / (std + 1e-8)
    num_train = train_normalized.shape[0]
    num_eval = eval_normalized.shape[0]
    num_features = train_normalized.shape[1] * train_normalized.shape[2]
    train_flat = train_normalized.reshape(num_train, num_features)
    eval_flat = eval_normalized.reshape(num_eval, num_features)
    dim_latent = cfg.model.dim_latent
    logger.info(f"Fitting PCA with {dim_latent} components on {num_train} samples")
    pca = PCA(n_components=dim_latent)
    pca.fit(train_flat)
    eval_reconstructed_flat = pca.inverse_transform(pca.transform(eval_flat))
    eval_reconstructed_normalized = eval_reconstructed_flat.reshape(eval_shapes.shape)
    eval_reconstructed = eval_reconstructed_normalized * (std + 1e-8) + mean
    errors = np.sqrt(np.sum((eval_shapes - eval_reconstructed) ** 2, axis=2)) * 1000.0
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    median_err = np.median(errors)
    greater_than_1_pct = np.mean(errors > 1.0)
    MetricEvent(
        model_name="PCA",
        source_name="Test",
        epoch=0,
        metrics={
            "error_mm_mean": float(mean_err),
            "error_mm_std": float(std_err),
            "error_mm_median": float(median_err),
            "error_mm_>1": float(greater_than_1_pct),
        },
    )
    return


def setup_and_train(cfg: AllConfig) -> None:
    """Set up experiment and start PCA fit."""
    trackers = get_trackers(cfg, debug=DEBUG_MODE)
    resume = cfg.user.load_checkpoint != 0
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, record=not DEBUG_MODE):
        train_pca()

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Entry point for training."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        from src.utils import DistributedWorker

        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)

    return


if __name__ == "__main__":
    main()
