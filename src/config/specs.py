"""Specification for the configuration files."""

import dataclasses
import pathlib

from typing import Any, Annotated

import torch

from pydantic import Field
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.config.environment import EnvSettings, VERSION
from src.config.torch import ActClass, get_activation_cls, get_optim_cls, set_seed
from src.config.options import (
    Datasets,
    ModelOperators,
    LossTypes,
    Schedulers,
)

PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


@dataclass
class DatasetConfig:
    """Specification for the dataset.

    Attributes:
        name (Datasets): The name of the dataset
    """

    name: Datasets


@dataclass
class DataConfig:
    """Specification for pre-processing the data.

    Attributes:
        dataset (DatasetConfig): The dataset configuration
    """

    dataset: DatasetConfig


@dataclass
class ModelConfig:
    """Specification for the VAE model architecture.

    Attributes:
        operator (ModelOperators): The name of the graph operator class
        n_features (StrictlyPositiveInt): The number of features in the VAE blocks
        dim_latent (StrictlyPositiveInt): The latent space dimension
        n_blocks_encoder (StrictlyPositiveInt): Number of graph blocks in the VAE encoder
        n_blocks_decoder (StrictlyPositiveInt): Number of graph blocks in the VAE decoder
        activation (str): The name of the PyTorch activation class (e.g., 'ELU')
        n_dense (StrictlyPositiveInt): The hidden number of features of the decoder for z
        use_mean_shape (bool): Whether to use the mean shape at the start of the decoder
    """

    operator: ModelOperators
    n_features: StrictlyPositiveInt = 64
    dim_latent: StrictlyPositiveInt = 8
    n_blocks_encoder: StrictlyPositiveInt = 2
    n_blocks_decoder: StrictlyPositiveInt = 3
    activation: str = "ELU"
    n_dense: StrictlyPositiveInt = 64
    use_mean_shape: bool = True

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        self.activation_cls: ActClass = get_activation_cls(self.activation)
        return


@dataclass
class SchedulerConfig:
    """Specification for the learning rate scheduler.

    Attributes:
        function (Schedulers): The name of the scheduler function
        restart_interval (PositiveInt): The number of epochs between restarts (0 disables restarts)
        restart_fraction (PositiveFloat): The fraction of the base learning rate when restarting (0.0 disables restarts)
        warmup_steps (PositiveInt): The number of initial epochs with linearly increasing learning rate (0 disables warmup)
        settings (dict): A dictionary containing default settings for the scheduler
    """

    function: Schedulers
    restart_interval: PositiveInt = 0
    restart_fraction: PositiveFloat = 0.0
    warmup_steps: PositiveInt = 0
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class LearningConfig:
    """Specification for the learning scheme.

    Attributes:
        optimizer_name (str): The name of the PyTorch optimizer (e.g., 'Adam', 'SGD')
        learning_rate (PositiveFloat): The base learning rate
        weight_decay (PositiveFloat): The weight decay parameter for the optimizer
        scheduler (SchedulerConfig): The scheduler configuration for learning rate decay
        opt_settings (dict): A dictionary containing default settings for the optimizer
    """

    optimizer_name: str
    learning_rate: PositiveFloat
    weight_decay: PositiveFloat
    scheduler: SchedulerConfig
    opt_settings: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve optimizer class from name."""
        self.optimizer_cls = get_optim_cls(self.optimizer_name)
        return


@dataclass
class EarlyStoppingConfig:
    """Specification for early stopping during training.

    Attributes:
        active (bool): Whether to use the early stopping strategy
        window (int): The number of last metrics to average
        patience (int): The number of epochs to wait before stopping
    """

    active: bool
    window: int = 1
    patience: int = 10


@dataclass
class TrainingConfig:
    """Specification for the training process.

    Attributes:
        batch_size (StrictlyPositiveInt): The batch size for training
        learn (LearningConfig): The learning configuration for training
        n_epochs (StrictlyPositiveInt): The total number of epochs for training
        early_stopping (EarlyStoppingConfig): The configuration for early stopping
        loss_type (LossTypes): The loss function type
        annealing_period (PositiveInt): Epoch period for KL divergence annealing (0 disables annealing)
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig
    loss_type: LossTypes

    annealing_period: PositiveInt = 100


@dataclass
class PathSpecs:
    """Path specifications for directories.

    Attributes:
        root_exp_dir (pathlib.Path): The directory containing the experiments
        data_dir (pathlib.Path): The directory containing the datasets (the folder where the COMA_data folder is located)
        metadata_dir (pathlib.Path): The directory containing the dataset metadata
    """

    _env = EnvSettings()
    root_exp_dir: pathlib.Path = _env.root_exp_dir
    data_dir: pathlib.Path = _env.dataset_dir
    metadata_dir: pathlib.Path = _env.metadata_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """The full path for the version directory."""
        return self.root_exp_dir / f"v{VERSION}"


@dataclass
class TrackerList:
    """List of trackers to use for logging.

    Attributes:
        wandb (bool): Whether to use the Wandb tracker
        hydra (bool): Whether to use the HydraLink tracker
        csv (bool): Whether to use the CSVDumper tracker
        tensorboard (bool): Whether to use the TensorBoard tracker
        sqlalchemy (bool): Whether to use the SQLAlchemyConnection tracker
    """

    wandb: bool
    hydra: bool
    csv: bool
    tensorboard: bool
    sqlalchemy: bool


@dataclasses.dataclass
class HydraSettings:
    """Subset of the current hydra settings.

    Attributes:
        output_dir (pathlib.Path): The Hydra run output directory
        job_logging (DictConfig): Logging configuration used by Hydra
    """

    output_dir: pathlib.Path = dataclasses.field(init=False)
    job_logging: DictConfig = dataclasses.field(init=False)


@dataclass
class UserSettings:
    """User-specific options and preferences.

    Attributes:
        cpu (bool): Whether to run computations on CPU (None/False defaults to accelerator)
        n_workers (PositiveInt): The number of workers for data loading
        trackers (TrackerList): The active trackers configurations
        seed (int | None): The seed for PyTorch/NumPy randomness (None disables manual seeding)
        checkpoint_every (PositiveInt): The number of epochs between saving checkpoints
        on_the_fly (bool): Whether to load/preprocess data on-the-fly instead of in-memory
        load_checkpoint (int): The checkpoint epoch to load (0 starts from scratch)
        n_subprocesses (PositiveInt): Number of subprocesses for distributed training (0 for no parallelism)
    """

    cpu: bool
    n_workers: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    on_the_fly: bool = False
    load_checkpoint: int = 0
    n_subprocesses: PositiveInt = 0
    hydra = HydraSettings()
    path = PathSpecs()

    def __post_init__(self) -> None:
        """Set seed for PyTorch and NumPy."""
        if self.seed is not None:
            set_seed(self.seed)

        return

    @property
    def device(self) -> torch.device | None:
        """The device where the model should run (None for default)."""
        return torch.device("cpu") if self.cpu else None


@dataclass
class AllConfig:
    """Root specification for all experiment settings.

    Attributes:
        variation (str): The name for the experiment
        final (bool): If True, it disables validation and extra logs for the final training run
        model (ModelConfig): VAE model architecture configuration
        user (UserSettings): User-specific settings
        data (DataConfig): Data pre-processing configuration
        train (TrainingConfig): Training options
        tags (list[str]): Descriptive tags for the experiment run
    """

    variation: str
    final: bool
    model: ModelConfig
    user: UserSettings
    data: DataConfig
    train: TrainingConfig
    tags: list[str] = dataclasses.field(default_factory=list)
    version = f"v{VERSION}"

    @property
    def name(self) -> str:
        """The full name of the experiment."""
        out = f"{self.variation}_final" if self.final else self.variation
        return out[:255]

    @property
    def project(self) -> str:
        """Project name for logging."""
        return "FixedGraphVariationalAutoEncoder" + str(self.version)
