"""Hydra configuration."""

import functools
import pathlib
from typing import Any, cast
from collections.abc import Callable

import hydra
from hydra.conf import HydraConf
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from pydantic import TypeAdapter

from src.config.environment import ConfigPath
from src.config.specs import AllConfig, HydraSettings
from src.config.experiment import get_variation


def hydra_main(func: Callable[[AllConfig], None]) -> Callable[[], None]:
    """Start hydra run and Converts dict_cfg to ConfigAll."""

    @hydra.main(
        version_base=None,
        config_path=str(ConfigPath.CONFIGS.absolute()),
        config_name='defaults',
    )
    @functools.wraps(func)
    def wrapper(dict_cfg: DictConfig) -> None:
        """Convert configuration to the stored object"""
        hydra_dict_cfg = HydraConfig.get()
        settings = get_hydra_settings(hydra_dict_cfg)
        return func(_validate(dict_cfg, settings, list(hydra_dict_cfg.overrides.task)))

    return wrapper.__call__


def get_config_all(overrides: list[str] | None = None) -> AllConfig:
    """Get hydra configuration without starting a run."""
    GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path=ConfigPath.CONFIGS.relative()):
        dict_cfg = hydra.compose(config_name='defaults', overrides=overrides)
        try:
            hydra_settings = get_hydra_settings(HydraConfig.get())
        except ValueError:
            hydra_settings = HydraSettings(job_logging=OmegaConf.create())

        return _validate(dict_cfg, hydra_settings, overrides)


def get_hydra_settings(dict_cfg: HydraConf) -> HydraSettings:
    """Get subset of hydra settings."""
    return HydraSettings(
        output_dir=pathlib.Path(dict_cfg.runtime.output_dir),
        job_logging=cast(DictConfig, dict_cfg.job_logging),
    )


def _validate(dict_cfg: DictConfig, settings: HydraSettings, overrides: list[str] | None) -> AllConfig:
    values = cast(dict[str, Any], OmegaConf.to_container(dict_cfg, resolve=True))
    values['user']['hydra'] = settings
    if overrides is not None:
        values['variation'], values['tags'] = get_variation(values.get('variation'), overrides)

    return TypeAdapter(AllConfig).validate_python(values, extra='forbid')
