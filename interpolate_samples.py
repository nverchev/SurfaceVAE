"""Interpolate latent vectors of test samples and render the generated meshes."""

import logging
import random

from collections.abc import Sized

import torch

from drytorch.lib import loading

from src.config import AllConfig, Experiment, MeshEvent, get_trackers, hydra_main
from src.data import get_active_dataset
from src.data.split import Partitions
from src.train.loaders import collate_coma_batch
from src.train.models import load_extract_vae_module


@torch.inference_mode()
def interpolate_samples() -> None:
    """Encode meshes, interpolate along latent dimensions, and render the results."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
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
    mean_operator_adjoint = vae_module.normalize_operator(vae_module.mean_operator_adjoint)
    inputs, targets = loading.take_from_dataset(
        dataset, indices=sample_indices, device=device, collate_fn=collate_coma_batch
    )
    latents = vae_module(inputs).z
    dim_latent = vae_module.dim_latent
    for k, (i, label_idx) in enumerate(zip(sample_indices, targets.label.tolist(), strict=True)):
        z = latents[k : k + 1]
        label_name = class_names[label_idx] if class_names else f'subject_{label_idx}'
        logging.info('Interpolating Sample %d (label: %s):', i, label_name)
        for d in range(dim_latent):
            steps = cfg_user.plot.interpolation_steps
            factor = cfg_user.plot.interpolation_step_factor
            for j in range(-steps, steps + 1):
                z_mod = z.clone()
                z_mod[0, d] = (1.0 + factor * j) * z_mod[0, d]
                recon_mu = vae_module.decode(z_mod, mean_operator, mean_operator_adjoint, decoder_mean_shape)
                recon_vertices = vae_module.denormalize_sample(recon_mu).squeeze(0).cpu().numpy()
                MeshEvent(
                    vertices=[recon_vertices],
                    faces=faces,
                    title=f'{label_name}_interpolate_dim{d}_step{j}',
                    folder=f'reconstructed/sample_{i}/interpolate_dim_{d}',
                )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch sample interpolation."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, record=False):
        interpolate_samples()

    return


if __name__ == '__main__':
    main()
