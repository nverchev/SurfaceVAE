"""Tour the latent space along a spherical trajectory and render the generated meshes."""

import logging

import numpy as np
import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data.coma import BaseCOMAData
from src.train.models import load_extract_vae_module
from src.utils.visualization import render_mesh


@torch.inference_mode()
def tour_latent_space() -> None:
    """Generate and render meshes along a closed curve on the latent sphere."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    interactive = cfg_user.plot.interactive
    save_dir = cfg.user.path.version_dir / "images" / cfg.name / "generated" / "tour"
    save_dir.mkdir(parents=True, exist_ok=True)
    vae_module = load_extract_vae_module()
    device = vae_module.mean_shape.device
    faces = BaseCOMAData.get_faces()
    if vae_module.use_mean_shape:
        decoder_mean_shape = vae_module.normalize_sample(vae_module.mean_shape)
    else:
        decoder_mean_shape = torch.ones_like(vae_module.mean_shape)

    mean_operator = vae_module.normalize_operator(vae_module.mean_operator)
    mean_operator_adjoint = vae_module.normalize_operator(
        vae_module.mean_operator_adjoint
    )
    n_steps_per_plane = cfg_user.generate.tour_samples_per_plane
    radius = cfg_user.generate.tour_radius
    d_latent = vae_module.dim_latent
    if d_latent < 2:
        raise ValueError(
            f"Latent space dimension must be at least 2 for touring, but got {d_latent}"
        )

    total_steps = (d_latent - 1) * n_steps_per_plane
    global_step = 0
    for i in range(1, d_latent):
        for j in range(n_steps_per_plane):
            t = 2.0 * np.pi * j / n_steps_per_plane
            v = np.zeros(d_latent, dtype=np.float32)
            v[0] = np.cos(t)
            v[i] = np.sin(t)
            z_np = radius * v
            z = (
                torch.from_numpy(z_np)
                .unsqueeze(0)
                .to(dtype=torch.float32, device=device)
            )
            recon_mu = vae_module.decode(
                z, mean_operator, mean_operator_adjoint, decoder_mean_shape
            )
            recon_vertices = (
                vae_module.denormalize_sample(recon_mu).squeeze(0).cpu().numpy()
            )
            logging.info(
                "Rendering Tour Step %d / %d (Plane %d/%d, Step %d/%d)",
                global_step + 1,
                total_steps,
                i,
                d_latent - 1,
                j + 1,
                n_steps_per_plane,
            )
            render_mesh(
                recon_vertices,
                faces,
                title=f"tour_step_{global_step:03d}",
                interactive=interactive,
                save_dir=save_dir,
            )
            global_step += 1

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch latent space touring."""
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    with exp.create_run(resume=True):
        tour_latent_space()

    return


if __name__ == "__main__":
    main()
