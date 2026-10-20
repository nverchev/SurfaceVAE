"""Main script for training the Fixed Graph Variational AutoEncoder."""

import logging
from typing import TYPE_CHECKING, Any

from drytorch import Test, Trainer

from drytorch.lib.objectives import join


from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers, save_learning_curves
from src.config.environment import DEBUG_MODE
from src.data import get_splits
from src.module import get_vae_module
from src.train import (
    VAEEMAModel,
    VAEModel,
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


logger = logging.getLogger(__name__)


def train_vae(trial: Trial | None = None) -> None:
    """Set up the experiment and launch the VAE training."""
    cfg = Experiment.get_config()
    train_ds, eval_ds = get_splits()
    train_loader = COMADataLoader(
        train_ds,
        batch_size=cfg.train.batch_size_per_device,
        n_workers=cfg.user.n_workers,
    )
    eval_loader = COMADataLoader(
        eval_ds,
        batch_size=cfg.train.batch_size_per_device,
        n_workers=cfg.user.n_workers,
    )
    vae = get_vae_module(
        mean_shape=train_ds.mean_shape,
        mean_operator=train_ds.mean_operator,
        mean_operator_adjoint=train_ds.mean_operator_adjoint,
    )
    if trial is None:
        n_params = sum(p.numel() for p in vae.parameters())
        trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        n_buffers = sum(b.numel() for b in vae.buffers())
        logger.info(f'Model parameters: {n_params:,} (trainable: {trainable_params:,}, buffers: {n_buffers:,})')
        model = VAEEMAModel(vae, name=f'VAE_{cfg.model.operator}', device=cfg.user.device)
    else:
        model = VAEModel(vae, device=cfg.user.device)

    loss = get_vae_loss(model, normalize=vae.normalize_sample)
    metrics = get_reconstruction_metrics(denormalize=vae.denormalize_sample)
    loss_with_metrics = join(loss, metrics)
    learning_schema = get_learning_schema(cfg)
    builder = Trainer.builder(
        model,
        loader=train_loader,
        loss=loss,
        learning_schema=learning_schema,
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    if not cfg.final and cfg.train.early_stopping.active:
        cfg_early = cfg.train.early_stopping
        register_early_stopping(builder, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(builder, trial)
    else:
        register_checkpointing(builder, checkpoint_every=cfg.user.checkpoint_every)

    trainer = builder.build()
    if cfg.final:
        evaluation = Test(
            model,
            loader=eval_loader,
            metric=loss_with_metrics,
        )
    else:
        evaluation = builder.validation

    if cfg.user.load != 0:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    evaluation()
    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Set up experiment and start VAE training loop."""
    trackers = get_trackers(cfg)
    resume = cfg.user.load != 0 or cfg.user.run_id is not None
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_vae()
        save_learning_curves(trackers, f'VAE_{cfg.model.operator}', show=cfg.user.plot.show)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for training."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        from src.utils import DistributedWorker

        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)

    return


if __name__ == '__main__':
    main()
