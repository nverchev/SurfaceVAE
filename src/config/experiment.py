"""Contains the Experiment class for the project and related helper functions."""

import os
import warnings

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core import tracking
from drytorch.trackers import logging

from src.config.environment import DEBUG_MODE, get_current_branch
from src.config.specs import AllConfig


class Experiment(drytorch.Experiment[AllConfig]):
    """Contains the specifications for the current experiment."""

    pass


def get_trackers(cfg: AllConfig, debug_mode: bool = DEBUG_MODE) -> list[tracking.Tracker]:
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

    if cfg_trackers.pyvista:
        try:
            from src.trackers.pyvista import MeshTracker

        except ImportError as err:
            warnings.warn(f'PyVista not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(MeshTracker(interactive=cfg.user.plot.interactive))

    return tracker_list


def get_variation(variation: object, overrides: list[str], /) -> tuple[str, list[str]]:
    """Add the overrides to the variation of the experiment.

    Args:
        variation: variation of the configuration, the current git branch when unset; the command
            line gives numbers as numbers.
        overrides: raw Hydra task override strings.

    Returns:
        The variation followed by the overrides, and the overrides as tags.
    """
    filtered_overrides = [
        override
        for override in overrides
        if override.split('.')[0] != 'user' and override.split('=')[0] not in ('final', 'variation')
    ]
    sorted_overrides = sorted(filtered_overrides)
    base = str(variation) if variation is not None else get_current_branch()
    name = '_'.join([base, *sorted_overrides]).replace('/', '_')
    return name, sorted_overrides


def set_tuning_logging() -> None:
    """Set the logging for tuning runs."""
    if 'PARALLEL_SEQ' in os.environ:
        drytorch.remove_all_default_trackers()
        drytorch.extend_default_trackers([logging.BuiltinLogger()])
        logging.set_verbosity(logging.INFO_LEVELS.test)
    else:
        drytorch.init_trackers(mode='minimal')

    return
