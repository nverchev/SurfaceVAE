"""Base / Abstract VAE class: BaseVAE and getters."""

from typing import cast

import math
import torch
import torch.nn as nn
import scipy.sparse as sparse

from src.config import Experiment, ModelOperators
from src.data import get_active_dataset, Input, Output, build_operator
from src.module.encoder import get_encoder
from src.module.decoder import get_decoder
from src.utils.sparse import scipy_sparse_to_pytorch_sparse


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
    mean_operator_adj_indices: torch.Tensor
    mean_operator_adj_values: torch.Tensor
    mean_operator_adj_size: torch.Size
    use_mean_shape: bool
    operator_type: ModelOperators
    recon_logvar: torch.Tensor

    def __init__(
        self,
        mean_shape: torch.Tensor,
        mean_operator: torch.Tensor | None,
        mean_operator_adjoint: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.dim_latent = cfg.model.dim_latent
        self.use_mean_shape = cfg.model.use_mean_shape
        self.operator_type = cfg.model.operator
        self.register_buffer(
            "recon_logvar",
            torch.full((mean_shape.size(0), 1), math.log(0.001)),
        )
        self.register_buffer("mean_shape", mean_shape)
        self.register_buffer("mean_shape_center", mean_shape.mean(dim=0, keepdim=True))
        self.register_buffer("mean_shape_std", mean_shape.std())
        if mean_operator is not None and mean_operator.numel() > 0:
            coalesced_op = mean_operator.coalesce()
            self.register_buffer("mean_operator_indices", coalesced_op.indices())
            self.register_buffer("mean_operator_values", coalesced_op.values())
            self.mean_operator_size = coalesced_op.size()
        else:
            self.register_buffer(
                "mean_operator_indices", torch.empty(0, dtype=torch.long)
            )
            self.register_buffer("mean_operator_values", torch.empty(0))
            self.mean_operator_size = torch.Size([0, 0])

        if mean_operator_adjoint is not None and mean_operator_adjoint.numel() > 0:
            coalesced_adj = mean_operator_adjoint.coalesce()
            self.register_buffer("mean_operator_adj_indices", coalesced_adj.indices())
            self.register_buffer("mean_operator_adj_values", coalesced_adj.values())
            self.mean_operator_adj_size = coalesced_adj.size()
        else:
            self.register_buffer(
                "mean_operator_adj_indices", torch.empty(0, dtype=torch.long)
            )
            self.register_buffer("mean_operator_adj_values", torch.empty(0))
            self.mean_operator_adj_size = torch.Size([0, 0])

        self.encoder = get_encoder()
        self.decoder = get_decoder()
        return

    @property
    def mean_operator(self) -> torch.Tensor:
        if self.mean_operator_indices.numel() == 0:
            return torch.empty(0)

        return torch.sparse_coo_tensor(
            self.mean_operator_indices,
            self.mean_operator_values,
            self.mean_operator_size,
        )

    @property
    def mean_operator_adjoint(self) -> torch.Tensor:
        if self.mean_operator_adj_indices.numel() == 0:
            return torch.empty(0)

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
        raw_inputs: torch.Tensor,
        operator: torch.Tensor | None,
        operator_adjoint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, raw_inputs, operator, operator_adjoint)

    def decode(
        self,
        z: torch.Tensor,
        mean_operator: torch.Tensor | None,
        mean_operator_adjoint: torch.Tensor | None,
        mean_shape: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(z, mean_operator, mean_operator_adjoint, mean_shape)

    def forward(self, inputs: Input) -> Output:
        out = Output()
        sample = self.normalize_sample(inputs.x)
        operator = self.normalize_operator(inputs.operator)
        operator_adjoint = self.normalize_operator(inputs.operator_adjoint)
        out.mu, out.logvar = self.encode(
            sample, inputs.x, operator, operator_adjoint
        ).chunk(2, dim=-1)
        out.z = self.sample(out.mu, out.logvar) if self.training else out.mu
        if self.use_mean_shape:
            decoder_mean_shape = self.normalize_sample(self.mean_shape)
        else:
            decoder_mean_shape = torch.ones_like(self.mean_shape)

        mean_operator = self.normalize_operator(self.mean_operator)
        mean_operator_adjoint = self.normalize_operator(self.mean_operator_adjoint)
        out.recon_mu = self.decode(
            out.z, mean_operator, mean_operator_adjoint, decoder_mean_shape
        )
        out.recon_logvar = self.recon_logvar
        return out

    def generate_sample(self, n_samples: int) -> torch.Tensor:
        out = Output()
        out.z = torch.randn(n_samples, self.dim_latent, device=self.mean_shape.device)
        if self.use_mean_shape:
            decoder_mean_shape = self.normalize_sample(self.mean_shape)
        else:
            decoder_mean_shape = torch.ones_like(self.mean_shape)

        mean_operator = self.normalize_operator(self.mean_operator)
        mean_operator_adjoint = self.normalize_operator(self.mean_operator_adjoint)
        recon_mu = self.decode(
            out.z, mean_operator, mean_operator_adjoint, decoder_mean_shape
        )
        return self.denormalize_sample(recon_mu)

    def normalize_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if self.use_mean_shape:
            return (sample - self.mean_shape_center) / (self.mean_shape_std + 1e-8)

        return (sample - self.mean_shape) / (self.mean_shape_std + 1e-8)

    def denormalize_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if self.use_mean_shape:
            return sample * self.mean_shape_std + self.mean_shape_center

        return sample * self.mean_shape_std + self.mean_shape

    def normalize_operator(self, operator: torch.Tensor | None) -> torch.Tensor | None:
        if operator is None or operator.numel() == 0:
            return None

        if self.operator_type == ModelOperators.lap_beltrami:
            return operator * (self.mean_shape_std**2)

        if self.operator_type == ModelOperators.dirac:
            return operator * self.mean_shape_std

        return operator


def compute_operators_on_the_fly(
    x: torch.Tensor, operator_type: ModelOperators
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.size(0)
    faces = get_active_dataset().faces
    computed_ops = []
    computed_adjs = []
    x_cpu = x.detach().cpu()
    for i in range(batch_size):
        v_np = x_cpu[i].numpy()
        op, op_adj = build_operator((v_np, faces), operator_type)
        computed_ops.append(op)
        if op_adj is not None:
            computed_adjs.append(op_adj)

    operator_scipy = cast(sparse.coo_matrix, sparse.block_diag(computed_ops).tocoo())
    operator = scipy_sparse_to_pytorch_sparse(operator_scipy).to(x.device)
    if computed_adjs:
        operator_adjoint_scipy = cast(
            sparse.coo_matrix, sparse.block_diag(computed_adjs).tocoo()
        )
        operator_adjoint = scipy_sparse_to_pytorch_sparse(operator_adjoint_scipy).to(
            x.device
        )
    else:
        operator_adjoint = torch.empty(0, device=x.device)

    return operator, operator_adjoint


def get_vae_module(
    mean_shape: torch.Tensor,
    mean_operator: torch.Tensor | None,
    mean_operator_adjoint: torch.Tensor | None = None,
) -> BaseVAE:
    """Get VAE module using the provided mean shape and operators."""
    return BaseVAE(
        mean_shape=mean_shape,
        mean_operator=mean_operator,
        mean_operator_adjoint=mean_operator_adjoint,
    )
