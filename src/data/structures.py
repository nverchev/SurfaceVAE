"""Module containing classes for inputs, outputs and targets for the VAE."""

import dataclasses
from typing import NamedTuple, Self

import torch


class Input(NamedTuple):
    """Input for the Fixed Graph VAE.

    Attributes:
        x: the input vertices (batch_size, n_vertices, 3).
        operator: the main operator (e.g. Laplacian, Dirac).
        operator_adjoint: the adjoint operator (optional).
    """

    x: torch.Tensor
    operator: torch.Tensor | None
    operator_adjoint: torch.Tensor | None = None


class Target(NamedTuple):
    """Targets for the Fixed Graph VAE (reconstruction target).

    Attributes:
        x: the target vertices (batch_size, n_vertices, 3).
        label: optional class/expression label.
    """

    x: torch.Tensor
    label: torch.Tensor | None = None


@dataclasses.dataclass(init=False, slots=True)
class Output:
    """Outputs for the Fixed Graph VAE.

    Attributes:
        model_epoch: the epoch of the model.
        recon_mu: reconstructed vertex mean.
        recon_logvar: reconstructed vertex log variance (only for some architectures, else empty).
        mu: latent space mean.
        logvar: latent space log variance.
        z: sampled latent code.
        target_residual: cached (targets.x - recon_mu) for reuse.
    """

    model_epoch: int
    recon_mu: torch.Tensor
    recon_logvar: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor

    def update(self, other: Self) -> None:
        """Update the state with another instance's one."""
        for attribute in other.__slots__:
            try:
                setattr(self, attribute, getattr(other, attribute))
            except AttributeError:
                pass
        return

    def __repr__(self) -> str:
        field_names = [f.name for f in dataclasses.fields(self) if f.repr]
        parts = []
        for name in field_names:
            if hasattr(self, name):
                try:
                    value = getattr(self, name)
                    parts.append(f"{name}={value!r}")
                except Exception:
                    parts.append(f"{name}=<error>")
            else:
                parts.append(f"{name}=<uninitialized>")
        return f"{self.__class__.__name__}({', '.join(parts)})"
