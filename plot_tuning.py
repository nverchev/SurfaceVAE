"""Visualize the hyperparameters' optimization results for the autoencoder."""

import hydra
import optuna
from omegaconf import DictConfig

from src.config import ConfigPath
from src.utils.tuning import get_study_name, visualize_study


@hydra.main(
    version_base=None,
    config_path=ConfigPath.TUNING_VAE.absolute(),
    config_name="defaults",
)
def plot_tuning(tune_cfg: DictConfig) -> None:
    """Load the Optuna study for the given stage and render visualizations.

    Args:
        tune_cfg: Hydra‑provided configuration containing the tuning
            parameters (storage, study name, overrides, and renderer).
    """
    study_name = get_study_name(f"{tune_cfg.tune.study_name}", tune_cfg.overrides)
    study = optuna.load_study(study_name=study_name, storage=tune_cfg.storage)
    visualize_study(study, tune_cfg.renderer)
    return


if __name__ == "__main__":
    plot_tuning()
