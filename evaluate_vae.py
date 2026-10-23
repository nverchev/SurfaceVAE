"""Evaluate the trained VAE on the evaluation partition.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_vae.py final=true
    uv run evaluate_vae.py data/dataset=bareteeth final=true
"""

from drytorch import Test
from drytorch.lib.objectives import join

from src.config import ExperimentConfig, Experiment, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import get_splits
from src.lib import COMADataLoader, VAEEMAModel, get_reconstruction_metrics, get_vae_loss
from src.lib.models import load_trained_state
from src.module import get_vae_module
from src.trackers import get_trackers


def evaluate_vae() -> None:
    """Test the loss and the reconstruction error of the trained VAE."""
    cfg = Experiment.get_config()
    train_ds, eval_ds = get_splits()
    eval_loader = COMADataLoader(eval_ds, batch_size=cfg.train.batch_size_per_device, n_workers=cfg.user.n_workers)
    vae = get_vae_module(
        mean_shape=train_ds.mean_shape,
        mean_operator=train_ds.mean_operator,
        mean_operator_adjoint=train_ds.mean_operator_adjoint,
    )
    model = VAEEMAModel(vae, name=f'VAE_{cfg.model.operator}', device=cfg.user.device)
    load_trained_state(model, epoch=cfg.user.load or -1)
    loss = get_vae_loss(model, normalize=vae.normalize_sample)
    metrics = get_reconstruction_metrics(denormalize=vae.denormalize_sample)
    Test(model, loader=eval_loader, metric=join(loss, metrics))()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the VAE."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_vae()

    return


if __name__ == '__main__':
    main()
