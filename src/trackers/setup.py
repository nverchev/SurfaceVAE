"""Setup of the trackers of the experiment."""

import os
import warnings

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core import tracking
from drytorch.trackers import logging

from src.config.environment import DEBUG_MODE
from src.config.specs import ExperimentConfig


__all__ = [
    'get_trackers',
    'set_tuning_logging',
]


def get_trackers(cfg: ExperimentConfig, debug_mode: bool = DEBUG_MODE) -> list[tracking.Tracker]:
    """Get trackers according to the user configuration."""
    cfg_trackers = cfg.user.trackers
    cfg_hydra = cfg.user.hydra
    hydra_utils.configure_log(cfg_hydra.job_logging)
    drytorch.init_trackers(mode='hydra')
    tracker_list: list[tracking.Tracker] = []
    if debug_mode:
        return tracker_list

    if cfg_trackers.wandb:
        try:
            import wandb

            from drytorch.trackers.wandb import Wandb

        except ImportError as err:
            warnings.warn(f'Wandb not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(Wandb(settings=wandb.Settings(project=cfg.project)))

    if cfg_trackers.hydra:
        from drytorch.trackers.hydra import HydraLink

        tracker_list.append(HydraLink(hydra_dir=cfg.user.hydra.output_dir))

    if cfg_trackers.csv:
        from drytorch.trackers.csv import CSVDumper

        tracker_list.append(CSVDumper())

    if cfg_trackers.tensorboard:
        try:
            from drytorch.trackers.tensorboard import TensorBoard

        except ImportError as err:
            warnings.warn(f'TensorBoard not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(TensorBoard())

    if cfg_trackers.sqlalchemy:
        try:
            import sqlalchemy

            from drytorch.trackers.sqlalchemy import SQLConnection

        except ImportError as err:
            warnings.warn(f'SQLAlchemy not installed, skipping. {err}', stacklevel=2)

        else:
            engine_path = cfg.user.path.version_dir / 'metrics.db'
            cfg.user.path.version_dir.mkdir(exist_ok=True)
            engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
            tracker_list.append(SQLConnection(engine=engine))

    if cfg_trackers.matplotlib:
        try:
            from drytorch.trackers.matplotlib import MatPlotter

        except ImportError as err:
            warnings.warn(f'Matplotlib not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(MatPlotter())

    if cfg_trackers.plotly:
        try:
            from drytorch.trackers.plotly import PlotlyPlotter

        except ImportError as err:
            warnings.warn(f'Plotly not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(PlotlyPlotter())

    if cfg_trackers.pyvista:
        try:
            from src.trackers.pyvista import MeshTracker

        except ImportError as err:
            warnings.warn(f'PyVista not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(MeshTracker(interactive=cfg.user.plot.interactive))

    return tracker_list


def set_tuning_logging() -> None:
    """Set the logging for tuning runs."""
    if 'PARALLEL_SEQ' in os.environ:
        drytorch.remove_all_default_trackers()
        drytorch.extend_default_trackers([logging.BuiltinLogger()])
        logging.set_verbosity(logging.INFO_LEVELS.test)
    else:
        drytorch.init_trackers(mode='minimal')

    return
