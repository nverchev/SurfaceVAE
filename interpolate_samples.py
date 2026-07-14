"""Interpolate latent vectors of test samples and render the generated meshes."""

import logging
import random

from collections.abc import Sized

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data import Input, get_active_dataset
from src.data.split import Partitions
from src.train.loaders import collate_coma_batch
from src.train.models import load_extract_vae_module
from src.utils.visualization import render_mesh


@torch.inference_mode()
def interpolate_samples() -> None:
    """Encode meshes, interpolate along latent dimensions, and render the results."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / "images" / cfg.name / "reconstructed"
    if cfg_user.plot.use_train:
        dataset = get_active_dataset().get_split(Partitions.train)
    else:
        dataset = get_active_dataset().get_split(Partitions.test)

    vae_module = load_extract_vae_module()
    device = vae_module.mean_shape.device
    class_names = dataset.class_names
    faces = dataset.faces
    sample_indices = cfg_user.plot.sample_indices
    if not sample_indices:
        assert isinstance(dataset, Sized)
        sample_indices = [random.randint(0, len(dataset) - 1)]

    if vae_module.use_mean_shape:
        decoder_mean_shape = vae_module.normalize_sample(vae_module.mean_shape)
    else:
        decoder_mean_shape = torch.ones_like(vae_module.mean_shape)

    mean_operator = vae_module.normalize_operator(vae_module.mean_operator)
    mean_operator_adjoint = vae_module.normalize_operator(
        vae_module.mean_operator_adjoint
    )
    for i in sample_indices:
        assert isinstance(dataset, Sized)
        if i >= len(dataset):
            raise ValueError(
                f"Index {i} is too large for the selected dataset of length {len(dataset)}"
            )

        item = dataset[i]
        inputs, targets = collate_coma_batch([item])
        x_dev = inputs.x.to(device)
        op_dev = inputs.operator.to(device) if inputs.operator is not None else None
        op_adj_dev = (
            inputs.operator_adjoint.to(device)
            if inputs.operator_adjoint is not None
            else None
        )
        batch_input = Input(x=x_dev, operator=op_dev, operator_adjoint=op_adj_dev)
        outputs = vae_module(batch_input)
        z = outputs.z
        dim_latent = vae_module.dim_latent
        label_idx = int(item[1].label.item()) if item[1].label is not None else 0
        label_name = class_names[label_idx] if class_names else f"subject_{label_idx}"
        logging.info("Interpolating Sample %d (label: %s):", i, label_name)
        for d in range(dim_latent):
            save_dir = save_dir_base / f"sample_{i}" / f"interpolate_dim_{d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            steps = cfg_user.plot.interpolation_steps
            factor = cfg_user.plot.interpolation_step_factor
            for j in range(-steps, steps + 1):
                z_mod = z.clone()
                z_mod[0, d] = (1.0 + factor * j) * z_mod[0, d]
                recon_mu = vae_module.decode(
                    z_mod, mean_operator, mean_operator_adjoint, decoder_mean_shape
                )
                recon_vertices = (
                    vae_module.denormalize_sample(recon_mu).squeeze(0).cpu().numpy()
                )
                render_mesh(
                    recon_vertices,
                    faces,
                    title=f"{label_name}_interpolate_dim{d}_step{j}",
                    interactive=interactive,
                    save_dir=save_dir,
                )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch sample interpolation."""
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    with exp.create_run(resume=True):
        interpolate_samples()

    return


if __name__ == "__main__":
    main()
