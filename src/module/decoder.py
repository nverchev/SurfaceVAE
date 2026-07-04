"""Decoder module containing BaseDecoder, LapDecoder, DirDecoder, and get_decoder."""

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from src.config import Experiment, ModelOperators
from src.module.layers import LinearLayer, DirResNet, LapResNet, PointwiseResNet
from src.data import N_FACES


class BaseDecoder(nn.Module, metaclass=ABCMeta):
    """Base Decoder abstract class."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.n_features = cfg.model.n_features
        self.n_dense = cfg.model.n_dense
        self.dim_latent = cfg.model.dim_latent
        self.act = cfg.model.activation_cls()
        self.momentum = cfg.model.batch_norm_momentum
        self.conv_shape = LinearLayer(
            3,
            self.n_features,
            use_batch_norm=False,
            act=nn.Hardtanh(-3.0, 3.0),
            batch_norm_momentum=self.momentum,
        )
        self.conv_latent = LinearLayer(
            self.n_features,
            self.n_features,
            use_batch_norm=True,
            batch_norm_momentum=self.momentum,
            act=nn.Hardtanh(-3.0, 3.0),
        )
        self.dense_latent1 = LinearLayer(
            self.dim_latent,
            self.n_dense,
            act=self.act,
            batch_norm_momentum=self.momentum,
        )
        self.dense_latent2 = LinearLayer(
            self.n_dense,
            self.n_features,
            truncated_init=True,
            batch_norm_momentum=self.momentum,
        )
        self.fc_mu = LinearLayer(
            self.n_features,
            3,
            use_batch_norm=True,
            truncated_init=True,
            batch_norm_momentum=self.momentum,
        )
        return

    def _prepare_latent(
        self, z: torch.Tensor, mean_shape: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_shape(mean_shape.unsqueeze(0)).expand(z.size(0), -1, -1)
        features = self.dense_latent1(z)
        features = self.dense_latent2(features)
        x = torch.mul(x, features.unsqueeze(1))
        x = self.conv_latent(x)
        return x

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.fc_mu(x)
        return mu

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor | None,
        mean_shape: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the decoder."""


class LapDecoder(BaseDecoder):
    """Standard Laplacian decoder."""

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
                for _ in range(cfg.model.n_blocks_decoder)
            ]
        )
        return

    def forward(
        self,
        inputs: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor | None,
        mean_shape: torch.Tensor,
    ) -> torch.Tensor:
        x = self._prepare_latent(inputs, mean_shape)
        for layer in self.layers:
            x = layer(operator, x)

        return self._forward_dense(x)


class DirDecoder(BaseDecoder):
    """Dirac decoder."""

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
                for _ in range(cfg.model.n_blocks_decoder)
            ]
        )
        return

    def forward(
        self,
        inputs: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor | None,
        mean_shape: torch.Tensor,
    ) -> torch.Tensor:
        assert operator_adjoint is not None
        x = self._prepare_latent(inputs, mean_shape)
        v = x
        f = torch.zeros(x.size(0), self.n_faces, self.n_features, device=inputs.device)
        for layer in self.layers:
            v, f = layer(operator, operator_adjoint, v, f)

        return self._forward_dense(v)


class PointNetDecoder(BaseDecoder):
    """PointNet decoder baseline."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.layers = nn.ModuleList(
            [
                PointwiseResNet(
                    cfg.model.n_features,
                    act=self.act,
                    batch_norm_momentum=self.momentum,
                )
                for _ in range(2 * cfg.model.n_blocks_decoder)
            ]
        )
        return

    def forward(
        self,
        inputs: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor | None,
        mean_shape: torch.Tensor,
    ) -> torch.Tensor:
        x = self._prepare_latent(inputs, mean_shape)
        for layer in self.layers:
            x = layer(x)

        return self._forward_dense(x)


def get_decoder() -> BaseDecoder:
    """Get the correct decoder based on configuration."""
    cfg = Experiment.get_config()
    operator_type = cfg.model.operator
    if operator_type == ModelOperators.none:
        return PointNetDecoder()

    if operator_type in (
        ModelOperators.dirac,
        ModelOperators.dirac_norm,
        ModelOperators.dirac_graph_norm,
    ):
        return DirDecoder()

    return LapDecoder()
