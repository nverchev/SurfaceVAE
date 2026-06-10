"""Tune the hyperparameters of the Fixed Graph VAE."""

import pathlib
from collections.abc import Callable

import hydra
import optuna
import torch
from omegaconf import DictConfig

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError

from src.config import ConfigPath, Experiment, get_config_all, set_tuning_logging
from src.utils.tuning import (
    get_study_name,
    impute_failed_trial,
    impute_pruned_trial,
    log_study_settings,
)
from train_vae import train_vae


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration."""

    def objective(trial: optuna.Trial) -> float:
        """Set up the experiment and launch the training cycle."""
        overrides = suggest_overrides(tune_cfg, trial)
        trial_cfg = get_config_all(overrides)
        trial_cfg.train.n_epochs = tune_cfg.n_epochs
        trial_cfg.user.seed = tune_cfg.seed
        exp = Experiment(
            trial_cfg,
            name=trial_cfg.name,
            par_dir=trial_cfg.user.path.version_dir,
            tags=trial_cfg.tags,
        )
        with exp.create_run(record=False):
            try:
                train_vae(trial=trial)
            except optuna.TrialPruned:
                return impute_pruned_trial(trial)

            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        res = get_final_value(trial)
        return res

    return objective


@hydra.main(
    version_base=None,
    config_path=ConfigPath.TUNING_VAE.absolute(),
    config_name="defaults",
)
def main(tune_cfg: DictConfig) -> None:
    """Set up the study and launch the optimization."""
    set_tuning_logging()
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    if tune_cfg.tune.use_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=tune_cfg.tune.n_startup_trials,
            n_warmup_steps=tune_cfg.tune.n_warmup_steps,
            interval_steps=tune_cfg.tune.interval_steps,
            n_min_trials=tune_cfg.tune.n_min_trials,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.GPSampler(warn_independent_sampling=False)
    study_name = get_study_name(tune_cfg.tune.study_name, tune_cfg.overrides)
    study = optuna.create_study(
        study_name=study_name,
        storage=tune_cfg.storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    log_study_settings(study, tune_cfg)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    return


if __name__ == "__main__":
    main()
