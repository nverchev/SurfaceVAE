"""Configuration relative to the environment."""

import enum
import pathlib
import shutil
import subprocess
import sys
import tomllib

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    'DEBUG_MODE',
    'PROJECT_NAME',
    'VERSION',
    'ConfigPath',
    'EnvSettings',
    'get_current_branch',
]

with open(pathlib.Path(__file__).resolve().parent.parent.parent / 'pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

PROJECT_NAME: str = pyproject['project']['name']
VERSION: str = pyproject['project']['version']

# a debugger runs the process: no data of the experiment is written
DEBUG_MODE: bool = any(module in sys.modules for module in ('pydevd', 'debugpy')) or sys.gettrace() is not None


class EnvSettings(BaseSettings):
    """Settings loaded from environment variables or the ``.env`` file.

    Attributes:
        data_dir: Root directory for the datasets, shared by all versions.
        metadata_dir: Root directory for the dataset metadata.
        root_exp_dir: Root directory for all experiment artefacts.
    """

    data_dir: pathlib.Path = pathlib.Path('./data')
    metadata_dir: pathlib.Path = pathlib.Path('./dataset_metadata')
    root_exp_dir: pathlib.Path = pathlib.Path('./experiments')
    model_config = SettingsConfigDict(env_file='.env')


class ConfigPath(enum.StrEnum):
    """Paths to configurations."""

    CONFIGS = 'experiment'
    TUNING_VAE = 'tuning/vae'

    @classmethod
    def get_folder(cls) -> str:
        """Return folder_name."""
        return 'configs'

    def get_path(self) -> pathlib.Path:
        """Return folder path."""
        return pathlib.Path(__file__).parent.parent.parent / self.get_folder() / self

    def absolute(self) -> str:
        """Absolute path to folder."""
        return str(self.get_path().absolute().resolve())

    def relative(self) -> str:
        """Relative path to folder."""
        return f'../../{self.get_folder()}/{self}'


def get_current_branch() -> str:
    """Get the current git branch name."""
    git_command = shutil.which('git')
    if git_command is None:
        raise ValueError('git not found in PATH.')

    try:
        result = subprocess.run(
            [git_command, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        msg = f'Failed to determine git branch: {e.stderr.strip()}'
        raise ValueError(msg) from e

    branch = result.stdout.strip()
    if branch == 'HEAD':
        msg = 'Detached HEAD state detected. Please checkout a branch.'
        raise ValueError(msg)

    return branch
