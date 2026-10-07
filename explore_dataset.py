"""Explore and visualize the 3D mesh dataset hierarchy."""

import logging
import random

from collections.abc import Sized

import torch

from src.config import AllConfig, Experiment, MeshEvent, get_trackers, hydra_main
from src.data import get_active_dataset
from src.data.split import Partitions


@torch.inference_mode()
def explore_dataset() -> None:
    """Visualize the point cloud/mesh dataset hierarchy."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if cfg_user.plot.use_train:
        dataset = get_active_dataset().get_split(Partitions.train)
    else:
        dataset = get_active_dataset().get_split(Partitions.test)

    class_names = dataset.class_names
    faces = dataset.faces
    sample_indices = cfg_user.plot.sample_indices
    if not sample_indices:
        assert isinstance(dataset, Sized)
        sample_indices = random.sample(range(len(dataset)), min(16, len(dataset)))

    for i in sample_indices:
        assert isinstance(dataset, Sized)
        if i >= len(dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(dataset)}')

        inputs, targets = dataset[i]
        label_idx = int(targets.label.item()) if targets.label is not None else 0
        label_name = class_names[label_idx] if class_names else f'subject_{label_idx}'
        logging.info('Exploring Dataset for Sample %d (label: %s):', i, label_name)
        vertices = inputs.x.numpy()
        MeshEvent(
            vertices=[vertices],
            faces=faces,
            title=f'{label_name}_mesh',
            folder=f'dataset/sample_{i}',
        )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the dataset exploration."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(record=False):
        explore_dataset()

    return


if __name__ == '__main__':
    main()
