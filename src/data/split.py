"""Module containing COMA dataset splits."""

import enum

from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.structures import Input, Target
from src.utils.sparse import scipy_sparse_to_pytorch_sparse
from collections.abc import Sequence
from src.data import operators
from src.config import Experiment, ModelOperators


class Partitions(enum.Enum):
    """Splits of the dataset."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class AbstractSplit(Dataset[tuple[Input, Target]], metaclass=ABCMeta):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Input, Target]: ...

    @property
    @abstractmethod
    def mean_shape(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def std_shape(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def mean_operator(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def mean_operator_adjoint(self) -> torch.Tensor | None: ...


class COMADatasetSplit(AbstractSplit):
    """COMADataset for a specific partition."""

    def __init__(
        self,
        shapes: np.ndarray,
        faces: np.ndarray,
        operator: Sequence[torch.Tensor] | None,
        adjoint_operator: Sequence[torch.Tensor] | None,
    ) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self._shapes = shapes
        self._faces = faces
        self._mean_shape = self.get_mean_shape()
        self._std_shape = self.get_std_shape()
        self._mean_operator, self._mean_operator_adjoint = self.get_mean_operators(
            cfg.model.operator
        )
        self._operators = operator
        self._adjoint_operators = adjoint_operator
        return

    @property
    def mean_shape(self) -> torch.Tensor:
        return torch.from_numpy(self._mean_shape).float()

    @property
    def std_shape(self) -> torch.Tensor:
        return torch.from_numpy(self._std_shape).float()

    @property
    def mean_operator(self) -> torch.Tensor:
        return self._mean_operator

    @property
    def mean_operator_adjoint(self) -> torch.Tensor | None:
        return self._mean_operator_adjoint

    def get_mean_shape(self) -> np.ndarray:
        return self._shapes.mean(axis=0)

    def get_std_shape(self) -> np.ndarray:
        return self._shapes.std(axis=0)

    def get_mean_operators(
        self, operator_type: ModelOperators
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean_operator, mean_operator_adjoint = operators.build_operator(
            (self._mean_shape, self._faces), operator_type
        )
        mean_operator_sparse = scipy_sparse_to_pytorch_sparse(mean_operator)
        if mean_operator_adjoint is not None:
            mean_operator_adjoint_sparse = scipy_sparse_to_pytorch_sparse(
                mean_operator_adjoint
            )
        else:
            mean_operator_adjoint_sparse = None

        return mean_operator_sparse, mean_operator_adjoint_sparse

    def __len__(self) -> int:
        return len(self._shapes)

    def __getitem__(self, index: int) -> tuple[Input, Target]:
        sample = torch.from_numpy(self._shapes[index])
        if self._operators is not None:
            operator = self._operators[index]
        else:
            operator = None

        if self._adjoint_operators is not None:
            operator_adjoint = self._adjoint_operators[index]
        else:
            operator_adjoint = None

        inputs = Input(x=sample, operator=operator, operator_adjoint=operator_adjoint)
        targets = Target(x=sample)
        return inputs, targets
