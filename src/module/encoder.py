"""Encoder module containing BaseEncoder, LapEncoder, DirEncoder, and get_encoder."""

import torch
import torch.nn as nn

from src.config import Experiment
from src.module.layers import LinearLayer, global_average, DirResNet, LapResNet
from src.data import N_FACES


class BaseEncoder(nn.Module):
    """Base Encoder abstract class."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.act = cfg.model.activation_cls()
        self.n_features = cfg.model.n_features
        self.dim_latent = cfg.model.dim_latent
        self.conv1 = LinearLayer(3, self.n_features, use_batch_norm=False)
        self.infer_latent = LinearLayer(
            self.n_features,
            2 * self.dim_latent,
            truncated_init=True,
        )
        return

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
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
                LapResNet(cfg.model.n_features, act=self.act)
                for _ in range(cfg.model.n_blocks_encoder)
            ]
        )
        return

    def forward(
        self,
        inputs: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor,
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
                DirResNet(cfg.model.n_features, act=self.act)
                for _ in range(cfg.model.n_blocks_encoder)
            ]
        )
        return

    def forward(
        self,
        inputs: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor,
    ) -> torch.Tensor:
        assert operator_adjoint.numel() > 0
        batch_size, _, _ = inputs.size()
        v = self.conv1(inputs)
        f = torch.zeros(batch_size, self.n_faces, self.n_features, device=inputs.device)
        for layer in self.layers:
            v, f = layer(operator, operator_adjoint, v, f)

        return self._forward_dense(v)


def get_encoder() -> BaseEncoder:
    """Get the correct encoder based on configuration."""
    cfg = Experiment.get_config()
    operator_type = cfg.model.operator
    from src.config.options import ModelOperators

    is_dirac = operator_type in (
        ModelOperators.dirac_norm,
        ModelOperators.dirac_graph_norm,
    )
    if is_dirac:
        return DirEncoder()

    return LapEncoder()
