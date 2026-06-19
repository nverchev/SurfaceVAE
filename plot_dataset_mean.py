"""Verification script to plot the dataset mean shape."""

import logging
import torch
from src.config import AllConfig, Experiment, hydra_main
from src.data import get_active_dataset
from src.data.split import Partitions
from src.utils.visualization import render_mesh


@torch.inference_mode()
def plot_dataset_mean() -> None:
    cfg = Experiment.get_config()
    interactive = cfg.user.plot.interactive
    dataset = get_active_dataset().get_split(Partitions.train)
    faces = dataset.faces
    mean_shape = dataset.mean_shape.numpy()
    save_dir = cfg.user.path.version_dir / "images" / "dataset"
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Rendering dataset mean shape...")
    render_mesh(
        ((mean_shape, faces),),
        title="dataset_mean_shape",
        interactive=interactive,
        save_dir=save_dir,
    )
    return


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    with exp.create_run(resume=True):
        plot_dataset_mean()
    return


if __name__ == "__main__":
    main()
