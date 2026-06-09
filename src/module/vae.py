"""Base / Abstract VAE class: BaseVAE and getters."""

import torch
import torch.nn as nn

from src.config import Experiment
from src.data.structures import Input, Output
from src.module.encoder import get_encoder
from src.module.decoder import get_decoder


class BaseVAE(nn.Module):
    """Base VAE class."""

    encoder: nn.Module
    decoder: nn.Module
    mean_shape: torch.Tensor
    mean_shape_center: torch.Tensor
    mean_shape_std: torch.Tensor
    mean_operator_indices: torch.Tensor
    mean_operator_values: torch.Tensor
    mean_operator_size: torch.Size
    mean_operator_adj_indices: torch.Tensor | None
    mean_operator_adj_values: torch.Tensor | None
    mean_operator_adj_size: torch.Size | None
    use_mean_shape: bool

    def __init__(
        self,
        mean_shape: torch.Tensor,
        mean_operator: torch.Tensor,
        mean_operator_adjoint: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.dim_latent = cfg.model.dim_latent
        self.use_mean_shape = cfg.model.use_mean_shape
        self.recon_logvar = nn.Parameter(torch.tensor([-5.0]))
        self.register_buffer("mean_shape", mean_shape)
        self.register_buffer("mean_shape_center", mean_shape.mean(dim=1, keepdim=True))
        self.register_buffer("mean_shape_std", mean_shape.std(dim=1, keepdim=True))
        coalesced_op = mean_operator.coalesce()
        self.register_buffer("mean_operator_indices", coalesced_op.indices())
        self.register_buffer("mean_operator_values", coalesced_op.values())
        self.mean_operator_size = coalesced_op.size()
        if mean_operator_adjoint is not None:
            coalesced_adj = mean_operator_adjoint.coalesce()
            self.register_buffer("mean_operator_adj_indices", coalesced_adj.indices())
            self.register_buffer("mean_operator_adj_values", coalesced_adj.values())
            self.mean_operator_adj_size = coalesced_adj.size()
        else:
            self.mean_operator_adj_indices = None
            self.mean_operator_adj_values = None
            self.mean_operator_adj_size = None

        self.encoder = get_encoder()
        self.decoder = get_decoder()
        return

    @property
    def mean_operator(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.mean_operator_indices,
            self.mean_operator_values,
            self.mean_operator_size,
        )

    @property
    def mean_operator_adjoint(self) -> torch.Tensor | None:
        if self.mean_operator_adj_indices is None:
            return None

        return torch.sparse_coo_tensor(
            self.mean_operator_adj_indices,
            self.mean_operator_adj_values,
            self.mean_operator_adj_size,
        )

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(
        self,
        x: torch.Tensor,
        operator: torch.Tensor,
        operator_adjoint: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x, operator, operator_adjoint)

    def decode(
        self,
        z: torch.Tensor,
        mean_operator: torch.Tensor | None,
        mean_operator_adjoint: torch.Tensor | None,
        mean_shape: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(z, mean_operator, mean_operator_adjoint, mean_shape)

    def forward(
        self,
        inputs: Input,
    ) -> Output:
        out = Output()
        op = inputs.operator
        op_adj = inputs.operator_adjoint
        sample = self.normalize_sample(inputs.x)
        out.mu, out.logvar = self.encode(sample, op, op_adj).chunk(2, dim=-1)
        out.z = self.sample(out.mu, out.logvar) if self.training else out.mu
        if self.use_mean_shape:
            decoder_mean_shape = self.mean_shape
        else:
            decoder_mean_shape = torch.zeros_like(self.mean_shape)

        out.recon_mu = self.decode(
            out.z,
            self.mean_operator,
            self.mean_operator_adjoint,
            decoder_mean_shape,
        )
        out.recon_logvar = self.recon_logvar
        return out

    def generate_sample(
        self,
        n_samples: int,
    ) -> Output:
        out = Output()
        out.z = torch.randn(n_samples, self.dim_latent, device=self.mean_shape.device)
        if self.use_mean_shape:
            decoder_mean_shape = self.mean_shape
        else:
            decoder_mean_shape = torch.zeros_like(self.mean_shape)

        out.recon_mu = self.decode(
            out.z,
            self.mean_operator,
            self.mean_operator_adjoint,
            decoder_mean_shape,
        )
        return out

    def normalize_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if self.use_mean_shape:
            return (sample - self.mean_shape) / (self.mean_shape_std + 1e-8)

        return (sample - self.mean_shape_center) / (self.mean_shape_std + 1e-8)

    def denormalize_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if self.use_mean_shape:
            return sample * self.mean_shape_std + self.mean_shape_center

        return sample * self.mean_shape_std + self.mean_shape


def get_vae_module(
    mean_shape: torch.Tensor,
    mean_operator: torch.Tensor,
    mean_operator_adjoint: torch.Tensor | None = None,
) -> BaseVAE:
    """Get VAE module using the provided mean shape and operators."""
    return BaseVAE(
        mean_shape=mean_shape,
        mean_operator=mean_operator,
        mean_operator_adjoint=mean_operator_adjoint,
    )
