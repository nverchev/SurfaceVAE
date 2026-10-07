"""Reconstruct samples from the dataset."""

import logging
import random

from collections.abc import Sized

import numpy as np
import torch

from src.config import AllConfig, Experiment, MeshEvent, get_trackers, hydra_main
from src.data import Inputs, get_active_dataset
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

    dataset = get_active_dataset().get_split(partition)

    vae_module = load_extract_vae_module()
    device = vae_module.mean_shape.device
    class_names = dataset.class_names
    faces = dataset.faces
    sample_indices = cfg_user.plot.sample_indices
    if not sample_indices:
        assert isinstance(dataset, Sized)
        sample_indices = random.sample(range(len(dataset)), min(8, len(dataset)))

    for i in sample_indices:
        assert isinstance(dataset, Sized)
        if i >= len(dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(dataset)}')

        item = dataset[i]
        inputs, _ = collate_coma_batch([item])
        x_dev = inputs.x.to(device)
        op_dev = inputs.operator.to(device) if inputs.operator is not None else None
        op_adj_dev = inputs.operator_adjoint.to(device) if inputs.operator_adjoint is not None else None
        batch_input = Inputs(x=x_dev, operator=op_dev, operator_adjoint=op_adj_dev)
        outputs = vae_module(batch_input)
        recon_mu = vae_module.denormalize_sample(outputs.recon_mu)
        recon_vertices = recon_mu.squeeze(0).cpu().numpy()
        original_vertices = item[0].x.numpy()
        label_idx = int(item[1].label.item()) if item[1].label is not None else 0
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
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the sample reconstruction."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        reconstruct_samples()

    return


if __name__ == '__main__':
    main()
