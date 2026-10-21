"""Verification script to plot the dataset mean shape."""

import logging
import torch
from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.lib.events import MeshEvent
from src.data import get_databuilder
from src.data.split import Partitions


@torch.inference_mode()
def plot_dataset_mean() -> None:
    split = get_databuilder().get_split(Partitions.train)
    faces = split.faces
    mean_shape = split.mean_shape.numpy()
    logging.info('Rendering dataset mean shape...')
    MeshEvent(
        vertices=[mean_shape],
        faces=faces,
        title='dataset_mean_shape',
        folder='dataset',
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        plot_dataset_mean()
    return


if __name__ == '__main__':
    main()
