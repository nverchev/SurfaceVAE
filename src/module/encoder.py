"""Encoder module containing BaseEncoder, LapEncoder, DirEncoder, and get_encoder."""

from abc import ABCMeta, abstractmethod

from typing import override

import torch
import torch.nn as nn

from src.config import Experiment, ModelOperators
from src.module.layers import (
    LinearLayer,
    global_average,
    DirResNet,
    LapResNet,
)
from src.data import N_FACES


class BaseEncoder(nn.Module, metaclass=ABCMeta):
    """Base Encoder abstract class."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.operator_type = cfg.model.operator
        self.act = cfg.model.activation_cls()
        self.n_features = cfg.model.n_features
        self.dim_latent = cfg.model.dim_latent
        self.momentum = cfg.model.batch_norm_momentum
        self.conv1 = LinearLayer(
            3,
            self.n_features,
            use_batch_norm=False,
            batch_norm_momentum=self.momentum,
        )
        self.infer_latent = LinearLayer(
            self.n_features,
            2 * self.dim_latent,
            truncated_init=True,
            batch_norm_momentum=self.momentum,
        )
        return

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        raw_inputs: torch.Tensor,
        operator: torch.Tensor | None,
        operator_adjoint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the encoder."""
        raise NotImplementedError()

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the dense layers."""
        x = global_average(x).squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.infer_latent(x)
        return x


class LapEncoder(BaseEncoder):
    """Standard Laplacian encoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.layers = nn.ModuleList(
            [
                LapResNet(
                    cfg.model.n_features,
                    act=self.act,
                    batch_norm_momentum=self.momentum,
                )
                for _ in range(cfg.model.n_blocks_encoder)
            ]
        )
        return

    @override
    def forward(
        self,
        inputs: torch.Tensor,
        raw_inputs: torch.Tensor,
        operator: torch.Tensor | None,
        operator_adjoint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv1(inputs)
        for layer in self.layers:
            x = layer(operator, x)

        return self._forward_dense(x)


class DirEncoder(BaseEncoder):
    """Dirac encoder."""

    def __init__(self, n_faces: int = N_FACES) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.n_features = cfg.model.n_features
        self.n_faces = n_faces
        self.layers = nn.ModuleList(
            [
                DirResNet(
                    cfg.model.n_features,
                    act=self.act,
                    batch_norm_momentum=self.momentum,
                )
                for _ in range(cfg.model.n_blocks_encoder)
            ]
        )
        return

    @override
    def forward(
        self,
        inputs: torch.Tensor,
        raw_inputs: torch.Tensor,
        operator: torch.Tensor | None,
        operator_adjoint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, _ = inputs.size()
        v = self.conv1(inputs)
        f = torch.zeros(batch_size, self.n_faces, self.n_features, device=inputs.device)
        for layer in self.layers:
            v, f = layer(operator, operator_adjoint, v, f)

        return self._forward_dense(v)


def get_encoder() -> BaseEncoder:
    """Get the correct encoder based on configuration."""
    cfg = Experiment.get_config()
    if cfg.model.operator in (
        ModelOperators.dirac,
        ModelOperators.dirac_stiff,
        ModelOperators.dirac_graph_norm,
    ):
        return DirEncoder()

    return LapEncoder()
