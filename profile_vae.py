"""Script for profiling the Fixed Graph Variational AutoEncoder."""

import logging
import pathlib

import torch
from drytorch.lib.loading import take_from_dataset

from src.config import ExperimentConfig, Experiment, hydra_main
from src.data import get_splits
from src.module import get_vae_module
from src.train.loaders import collate_coma_batch
from src.utils.profiling import log_parameter_counts, profile_forward, profile_backward

logger = logging.getLogger(__name__)


def run_profiling(cfg: ExperimentConfig) -> None:
    """Set up the split and model, then run profiling."""
    device = cfg.user.device
    if device == 'auto':
        device = torch.accelerator.current_accelerator(check_available=True) or torch.device('cpu')

    train_split, _ = get_splits()
    vae = get_vae_module(
        mean_shape=train_split.mean_shape,
        mean_operator=train_split.mean_operator,
        mean_operator_adjoint=train_split.mean_operator_adjoint,
    )
    profile_dir = _get_profiling_dir()
    batch_size = cfg.train.batch_size
    _, (inputs, _) = take_from_dataset(train_split, collate_fn=collate_coma_batch, device=device, n_samples=batch_size)
    log_parameter_counts(vae, profile_dir)
    vae.train()
    vae.to(device)
    for _ in range(3):
        outputs = vae(inputs)
        loss = outputs.recon_mu.sum()
        loss.backward()
        vae.zero_grad()

    outputs = profile_forward(vae, inputs, device, profile_dir)
    loss = outputs.recon_mu.sum()
    profile_backward(loss, device, profile_dir)
    return


def setup_and_profile(cfg: ExperimentConfig) -> None:
    """Set up experiment and start VAE profiling."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        run_profiling(cfg)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for profiling."""
    setup_and_profile(cfg)
    return


def _get_profiling_dir() -> pathlib.Path:
    exp = Experiment.get_current()
    if '@' in exp.run.id:
        day, time = exp.run.id.split('@')
        run_subdir = pathlib.Path(day) / time

    else:
        run_subdir = pathlib.Path(exp.run.id)

    profile_dir = exp.par_dir / 'profile' / exp.name / run_subdir
    profile_dir.mkdir(exist_ok=True, parents=True)
    return profile_dir


if __name__ == '__main__':
    main()
