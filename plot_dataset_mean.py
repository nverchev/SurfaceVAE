"""Verification script to plot the dataset mean shape."""

import logging
import torch
from src.config import AllConfig, Experiment, MeshEvent, get_trackers, hydra_main
from src.data import get_active_dataset
from src.data.split import Partitions


@torch.inference_mode()
def plot_dataset_mean() -> None:
    dataset = get_active_dataset().get_split(Partitions.train)
    faces = dataset.faces
    mean_shape = dataset.mean_shape.numpy()
    logging.info('Rendering dataset mean shape...')
    MeshEvent(
        vertices=[mean_shape],
        faces=faces,
        title='dataset_mean_shape',
        folder='dataset',
    )
    return


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        plot_dataset_mean()
    return


if __name__ == '__main__':
    main()
