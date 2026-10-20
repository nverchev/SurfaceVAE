"""Explore and visualize the 3D mesh dataset hierarchy."""

import logging

import torch

from drytorch.lib import loading

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.train.events import MeshEvent
from src.data import get_databuilder
from src.data.split import Partitions
from src.train.loaders import collate_coma_batch


@torch.inference_mode()
def explore_dataset() -> None:
    """Visualize the point cloud/mesh dataset hierarchy."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if cfg_user.visualization.use_train:
        split = get_databuilder().get_split(Partitions.train)
    else:
        split = get_databuilder().get_split(Partitions.test)

    class_names = split.class_names
    faces = split.faces
    if cfg_user.visualization.sample_indices:
        sample_indices, (inputs, targets) = loading.take_from_dataset(
            split, indices=cfg_user.visualization.sample_indices, collate_fn=collate_coma_batch
        )
    else:
        sample_indices, (inputs, targets) = loading.take_from_dataset(
            split, n_samples=cfg_user.generate.batch_size, preserve_order=False, collate_fn=collate_coma_batch
        )

    for i, sample_vertices, label in zip(sample_indices, inputs.x.unbind(), targets.label.unbind(), strict=True):
        vertices = sample_vertices.numpy()
        label_idx = int(label.item())
        label_name = class_names[label_idx] if class_names else f'subject_{label_idx}'
        logging.info('Exploring Dataset for Sample %d (label: %s):', i, label_name)
        MeshEvent(
            vertices=[vertices],
            faces=faces,
            title=f'{label_name}_mesh',
            folder=f'dataset/sample_{i}',
        )

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the dataset exploration."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(record=False):
        explore_dataset()

    return


if __name__ == '__main__':
    main()
