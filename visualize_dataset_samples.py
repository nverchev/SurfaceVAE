"""Visualise a few mesh samples from th dataset."""

import matplotlib.pyplot as plt

from src.config import AllConfig, Experiment, hydra_main
from src.data import get_splits


def visualise_samples() -> None:
    """Render dataset samples."""
    train_split, val_split = get_splits()
    indices = [0, 1, 2, 3]
    max_idx = len(train_split)
    indices = [i for i in indices if 0 <= i <= max_idx]
    n = len(indices)
    fig = plt.figure(figsize=(n * 3, 3))
    for i, idx in enumerate(indices, start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        verts = train_split[idx][0].x.numpy()
        ax.scatter(verts[:, 0], verts[:, 1], zs=verts[:, 2], s=1, c="steelblue")  # type: ignore
        ax.set_title(f"Sample {idx}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    return


def setup_and_visualize(cfg: AllConfig) -> None:
    """Set up experiment and visualize dataset samples."""
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )

    with exp.create_run(resume=False, record=False):
        visualise_samples()

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Entry point for visualizing dataset samples."""
    setup_and_visualize(cfg)
    return


if __name__ == "__main__":
    main()
