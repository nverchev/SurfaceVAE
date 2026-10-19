"""Reconstruct samples from the dataset."""

import logging

import numpy as np
import torch

from drytorch.lib import loading

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.train.events import MeshEvent
from src.data import get_databuilder
from src.data.split import Partitions
from src.train.loaders import collate_coma_batch
from src.train.models import load_extract_vae_module

ERROR_THRESHOLD = 10.0
CLIM_MAX = 5.0
CMAP = 'YlOrBr'
SHOW_SCALAR_BAR = False


@torch.inference_mode()
def reconstruct_samples() -> None:
    """Reconstruct and visualize selected dataset samples."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    split = get_databuilder().get_split(partition)
    vae_module = load_extract_vae_module()
    device = vae_module.mean_shape.device
    class_names = split.class_names
    faces = split.faces
    if cfg_user.plot.sample_indices:
        sample_indices, (inputs, targets) = loading.take_from_dataset(
            split, indices=cfg_user.plot.sample_indices, device=device, collate_fn=collate_coma_batch
        )
    else:
        sample_indices, (inputs, targets) = loading.take_from_dataset(
            split,
            n_samples=cfg_user.generate.batch_size,
            preserve_order=False,
            device=device,
            collate_fn=collate_coma_batch,
        )

    outputs = vae_module(inputs)
    originals = inputs.x.cpu().unbind()
    reconstructions = vae_module.denormalize_sample(outputs.recon_mu).cpu().unbind()
    for i, original, reconstruction, label in zip(
        sample_indices, originals, reconstructions, targets.label.unbind(), strict=True
    ):
        original_vertices = original.numpy()
        recon_vertices = reconstruction.numpy()
        label_idx = int(label.item())
        label_name = class_names[label_idx] if class_names else f'subject_{label_idx}'
        logging.info('Reconstructing Sample %d (label: %s):', i, label_name)
        folder = f'reconstructed/sample_{i}'
        MeshEvent(vertices=[original_vertices], faces=faces, title=f'{label_name}_original', folder=folder)
        MeshEvent(vertices=[recon_vertices], faces=faces, title=f'{label_name}_reconstructed', folder=folder)
        errors = np.linalg.norm(original_vertices - recon_vertices, axis=-1) * 1000.0
        errors = np.clip(errors, 0.0, ERROR_THRESHOLD)
        MeshEvent(
            vertices=[original_vertices],
            faces=faces,
            title=f'{label_name}_comparison',
            folder=folder,
            scalars=errors,
            cmap=CMAP,
            clim=[0.0, CLIM_MAX],
            show_scalar_bar=SHOW_SCALAR_BAR,
        )

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the sample reconstruction."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        reconstruct_samples()

    return


if __name__ == '__main__':
    main()
