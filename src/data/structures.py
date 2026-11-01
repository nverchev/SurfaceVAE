"""Module containing classes for inputs, outputs and targets for the VAE."""

import dataclasses
from typing import NamedTuple

import torch


class Inputs(NamedTuple):
    """Inputs for the Fixed Graph VAE.

    Attributes:
        x: the input vertices (batch_size, n_vertices, 3).
        operator: the main operator (e.g. Laplacian, Dirac).
        operator_adjoint: the adjoint operator (optional).
    """

    x: torch.Tensor
    operator: torch.Tensor | None
    operator_adjoint: torch.Tensor | None = None


class Targets(NamedTuple):
    """Targets for the Fixed Graph VAE (reconstruction target).

    Attributes:
        x: the target vertices (batch_size, n_vertices, 3).
        label: the label of the subject.
    """

    x: torch.Tensor
    label: torch.Tensor


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs for the Fixed Graph VAE.

    Attributes:
        recon_mu: reconstructed vertex mean.
        recon_logvar: reconstructed vertex log variance (only for some architectures, else empty).
        mu: latent space mean.
        logvar: latent space log variance.
        z: sampled latent code.
    """

    recon_mu: torch.Tensor
    recon_logvar: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
