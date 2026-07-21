"""Generate random samples using the VAE and render them."""

import logging

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data import get_active_dataset
from src.train.models import load_extract_vae_module
from src.utils.visualization import render_mesh


@torch.inference_mode()
def generate_samples() -> None:
    """Generate and visualize random mesh samples from the latent space."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    interactive = cfg_user.plot.interactive
    batch_size = cfg_user.generate.batch_size
    save_dir = cfg.user.path.version_dir / "images" / cfg.name / "generated"
    save_dir.mkdir(parents=True, exist_ok=True)
    faces = get_active_dataset().get_faces()
    vae_module = load_extract_vae_module()
    logging.info("Generating %d random samples using the VAE...", batch_size)
    generated_shapes = vae_module.generate_sample(batch_size)
    for i in range(batch_size):
        shape = generated_shapes[i].cpu().numpy()
        logging.info("  Rendering generated sample %d", i)
        render_mesh(
            shape,
            faces,
            title=f"generated_sample_{i}",
            interactive=interactive,
            save_dir=save_dir,
        )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the sample generation."""
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    with exp.create_run(resume=True):
        generate_samples()

    return


if __name__ == "__main__":
    main()
