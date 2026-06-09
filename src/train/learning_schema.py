"""Module for configuring the learning schema and scheduler."""

from drytorch import LearningSchema
from drytorch.lib import schedulers

from src.config.options import Schedulers
from src.config.specs import AllConfig, SchedulerConfig


def get_scheduler(config: SchedulerConfig) -> schedulers.AbstractScheduler:
    """Return the scheduler instance based on config."""
    if config.function == Schedulers.constant:
        scheduler: schedulers.AbstractScheduler = schedulers.ConstantScheduler()
    elif config.function == Schedulers.cosine:
        scheduler = schedulers.CosineScheduler(**config.settings)
    elif config.function == Schedulers.exponential:
        scheduler = schedulers.ExponentialScheduler(**config.settings)
    else:
        raise ValueError(f"Scheduler {config.function} not supported.")

    restart = schedulers.restart(
        restart_interval=config.restart_interval,
        restart_fraction=config.restart_fraction,
    )
    warmup = schedulers.warmup(config.warmup_steps)
    return scheduler.bind(warmup).bind(restart)


def get_learning_schema(cfg: AllConfig) -> LearningSchema:
    """Return configured learning scheme for training."""
    config = cfg.train.learn
    opt_defaults = {"weight_decay": config.weight_decay, **config.opt_settings}
    return LearningSchema(
        optimizer_cls=config.optimizer_cls,
        base_lr=config.learning_rate,
        scheduler=get_scheduler(config.scheduler),
        optimizer_defaults=opt_defaults,
    )
