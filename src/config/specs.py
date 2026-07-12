"""Specification for the configuration files."""

import dataclasses
import pathlib

from functools import cached_property
from typing import Any, Annotated, Self

import torch

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.config.environment import EnvSettings, VERSION, get_current_branch
from src.config.torch import ActClass, get_activation_cls, get_optim_cls, set_seed
from src.config.options import (
    Datasets,
    Expressions,
    Identities,
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
        validation (str | None): The validation expression or subject identity
        test (str | None): The test expression or subject identity
    """

    name: Datasets
    validation: str | None = None
    test: str | None = None

    @model_validator(mode="after")
    def _validate_extrapolation(self) -> Self:
        if self.name == Datasets.COMA_EXTRAPOLATION:
            try:
                self.validation = Expressions(self.validation)
                self.test = Expressions(self.test)
            except ValueError as e:
                msg = "Test and validation expressions must be specified for COMA_EXTRAPOLATION."
                raise ValueError(msg) from e

        if self.name == Datasets.COMA_IDENTITY:
            try:
                self.validation = Identities(self.validation)
                self.test = Identities(self.test)
            except ValueError as e:
                msg = "Both validation and test identities must be specified for COMA_IDENTITY."
                raise ValueError(msg) from e

        return self


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
        batch_norm_momentum (PositiveFloat | None): The momentum parameter for Batch Normalization layers
    """

    operator: ModelOperators
    n_features: StrictlyPositiveInt = 64
    dim_latent: StrictlyPositiveInt = 8
    n_blocks_encoder: StrictlyPositiveInt = 2
    n_blocks_decoder: StrictlyPositiveInt = 3
    activation: str = "ELU"
    n_dense: StrictlyPositiveInt = 64
    use_mean_shape: bool = True
    batch_norm_momentum: PositiveFloat | None = 0.1

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
class ObjectiveConfig:
    """Specification for the objective function.

    Attributes:
        beta (PositiveFloat): The Laplace loss scale parameter beta.
    """

    beta: PositiveFloat = 0.01


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
        objective (ObjectiveConfig): The objective configuration
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig
    loss_type: LossTypes
    _n_subprocesses: PositiveInt
    objective: ObjectiveConfig = dataclasses.field(default_factory=ObjectiveConfig)

    annealing_period: PositiveInt = 100

    @model_validator(mode="after")
    def _check_batch_size(self) -> Self:
        if self._n_subprocesses and self.batch_size % self._n_subprocesses != 0:
            msg = "Global batch size {} not divisible by number of devices {}."
            raise ValueError(msg.format(self.batch_size, self._n_subprocesses))

        return self

    @property
    def batch_size_per_device(self) -> int:
        """The batch size per device."""
        if self._n_subprocesses == 0:
            return self.batch_size

        return self.batch_size // self._n_subprocesses


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
class PlottingOptions:
    """Options for plotting and visualization."""

    interactive: bool
    sample_indices: list[PositiveInt] = dataclasses.field(default_factory=list)
    use_train: bool = False
    interpolation_step_factor: float = 0.3
    interpolation_steps: PositiveInt = 4
    num_eigenvectors: PositiveInt = 5


@dataclass
class GenerationOptions:
    """Options for the generation of meshes."""

    batch_size: StrictlyPositiveInt
    bias_dim: PositiveInt
    bias_value: float
    tour_samples_per_plane: StrictlyPositiveInt = 16
    tour_radius: float = 1.0


@dataclass
class UserSettings:
    """User-specific options and preferences.

    Attributes:
        cpu (bool): Whether to run computations on CPU (None/False defaults to accelerator)
        n_workers (PositiveInt): The number of workers for data loading
        trackers (TrackerList): The active trackers configurations
        seed (int | None): The seed for PyTorch/NumPy randomness (None disables manual seeding)
        checkpoint_every (PositiveInt): The number of epochs between saving checkpoints
        on_the_fly (bool): Compute operators on-the-fly per batch instead of loading from cache
        load_checkpoint (int): The checkpoint epoch to load (0 starts from scratch)
        n_subprocesses (PositiveInt): Number of subprocesses for distributed training (0 for no parallelism)
        plot (PlottingOptions): The plotting options
        generate (GenerationOptions): The generation options
    """

    cpu: bool
    n_workers: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    plot: PlottingOptions
    generate: GenerationOptions
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

    final: bool
    model: ModelConfig
    user: UserSettings
    data: DataConfig
    train: TrainingConfig
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = f"v{VERSION}"
    variation: str | None = None

    @cached_property
    def git_branch(self) -> str:
        """The current git branch.

        Raises:
            ValueError: If git is not available or if HEAD is detached.
        """
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Please specify 'variation' in the config."
            raise ValueError(msg) from ve

    @property
    def name(self) -> str:
        """The full name of the experiment."""
        return (
            f"{self.resolved_variation}_final"
            if self.final
            else self.resolved_variation
        )

    @property
    def project(self) -> str:
        """Project name for logging."""
        return "FixedGraphVariationalAutoEncoder" + str(self.version)

    @property
    def resolved_variation(self) -> str:
        """The resolved variation."""
        return self.variation or self.git_branch
