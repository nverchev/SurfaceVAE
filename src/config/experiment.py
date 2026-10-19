"""Contains the Experiment class for the project and related helper functions."""

import drytorch


from src.config.environment import get_current_branch
from src.config.specs import ExperimentConfig


class Experiment(drytorch.Experiment[ExperimentConfig]):
    """Contains the specifications for the current experiment."""

    pass


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
