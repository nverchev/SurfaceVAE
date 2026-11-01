"""Trackers handling the events of this project.

The trackers of optional dependencies live in their own modules, imported only
where the dependency is installed.
"""

from src.trackers.setup import get_trackers, set_tuning_logging


__all__ = [
    'get_trackers',
    'set_tuning_logging',
]
