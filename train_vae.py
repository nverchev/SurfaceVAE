"""Main script for training the Fixed Graph Variational AutoEncoder."""

import logging
import sys
from typing import TYPE_CHECKING, Any

from drytorch import Test, Trainer

from drytorch.lib import objectives

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.data import get_splits
from src.module import get_vae_module
from src.train import (
    ModelEpoch,
    get_learning_schema,
    COMADataLoader,
    get_vae_loss,
    get_reconstruction_metrics,
    register_checkpointing,
    register_early_stopping,
    register_pruning,
)

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


DEBUG_MODE = (
    any(module in sys.modules for module in ("pydevd", "debugpy", "pdb"))
    or sys.gettrace() is not None
)

logger = logging.getLogger(__name__)


def train_vae(trial: Trial | None = None) -> None:
    """Set up the experiment and launch the VAE training."""
    cfg = Experiment.get_config()
    train_ds, eval_ds = get_splits()
    train_loader = COMADataLoader(
        train_ds, batch_size=cfg.train.batch_size, n_workers=cfg.user.n_workers
    )
    eval_loader = COMADataLoader(
        eval_ds, batch_size=cfg.train.batch_size, n_workers=cfg.user.n_workers
    )
    vae = get_vae_module(
        mean_shape=train_ds.mean_shape,
        mean_operator=train_ds.mean_operator,
        mean_operator_adjoint=train_ds.mean_operator_adjoint,
    )
    if trial is None:
        n_params = sum(p.numel() for p in vae.parameters())
        trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,} (trainable: {trainable_params:,})")

    model = ModelEpoch(vae, name=f"VAE_{cfg.model.operator}", device=cfg.user.device)
    loss = get_vae_loss(normalize=vae.normalize_sample)
    metrics = get_reconstruction_metrics(denormalize=vae.denormalize_sample)
    loss_with_metrics = objectives.JoinLossMetrics(loss, metrics)
    learning_schema = get_learning_schema(cfg)
    trainer = Trainer(
        model,
        loader=train_loader,
        loss=loss,
        learning_schema=learning_schema,
    )

    test = Test(model, loader=eval_loader, metric=loss_with_metrics)
    if cfg.user.load_checkpoint != 0:
        trainer.load_checkpoint(cfg.user.load_checkpoint)

    if not cfg.final:
        trainer.add_validation(eval_loader)

    if not cfg.final and cfg.train.early_stopping.active:
        cfg_early = cfg.train.early_stopping
        register_early_stopping(
            trainer, window=cfg_early.window, patience=cfg_early.patience
        )

    if trial is not None:
        register_pruning(trainer, trial)
    else:
        register_checkpointing(trainer, cfg.user.checkpoint_every or 1)

    trainer.train_until(cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    test()
    return


def setup_and_train(cfg: AllConfig) -> None:
    """Set up experiment and start VAE training loop."""
    trackers = get_trackers(cfg, debug=DEBUG_MODE)
    resume = cfg.user.load_checkpoint != 0
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, record=not DEBUG_MODE):
        train_vae()

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
