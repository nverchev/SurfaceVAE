"""Generate random samples using the VAE and render them."""

import logging

import torch

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.train.events import MeshEvent
from src.data import get_databuilder
from src.train.models import load_extract_vae_module


@torch.inference_mode()
def generate_samples() -> None:
    """Generate and visualize random mesh samples from the latent space."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    batch_size = cfg_user.generate.batch_size
    z_max = cfg_user.generate.z_max
    z_std = cfg_user.generate.z_std
    faces = get_databuilder().get_faces()
    vae_module = load_extract_vae_module()
    logging.info(
        'Generating %d random samples using the VAE (z_max=%s, z_std=%s)...',
        batch_size,
        z_max,
        z_std,
    )
    generated_shapes = vae_module.generate_sample(batch_size, z_max=z_max, z_std=z_std)
    for i in range(batch_size):
        shape = generated_shapes[i].cpu().numpy()
        logging.info('  Rendering generated sample %d', i)
        MeshEvent(
            vertices=[shape],
            faces=faces,
            title=f'generated_sample_{i}',
            folder='generated',
        )

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the sample generation."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        generate_samples()

    return


if __name__ == '__main__':
    main()
