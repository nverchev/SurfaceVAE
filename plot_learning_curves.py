"""Plot the learning curves of a finished run from the metrics the trackers stored.

The plotting trackers read the metrics from the CSV files of the run, so neither the data
nor the model is needed.

Usage::

    uv run plot_learning_curves.py data/dataset=mouth_up final=true
    uv run plot_learning_curves.py user.run_id=2026-07-14@15h02m08s user.plot.show=true
"""

from drytorch.trackers import base_classes

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Resume the run and save the learning curves of its VAE."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    model_name = f'VAE_{cfg.model.operator}'
    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        for tracker in trackers:
            if isinstance(tracker, base_classes.BasePlotter):
                tracker.save_plots(model_name)
                if cfg.user.plot.show:
                    tracker.plot(model_name)

    return


if __name__ == '__main__':
    main()
