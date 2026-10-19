"""Loss functions and metrics for VAE training."""

from __future__ import annotations

import math
from typing import Any, Self, override
from collections.abc import Callable

import torch

from drytorch.core import protocols as p
from drytorch.lib import accumulators
from drytorch.lib.objectives import (
    Loss,
    LossWithMetrics,
    Objective,
)
from src.config.experiment import Experiment
from src.data.structures import Outputs, Targets


class ReconstructionErrorCompiler(accumulators.AbstractAccumulator[torch.Tensor, torch.Tensor]):
    """Accumulates the distances of every sample of an evaluation."""

    def __init__(self, values: list[torch.Tensor]) -> None:
        self.values = values
        return

    @classmethod
    @override
    def from_value(cls, value: torch.Tensor) -> Self:
        return cls([value.detach()])

    @override
    def merge(self, other: ReconstructionErrorCompiler) -> None:
        self.values.extend(other.values)
        return

    @override
    def sync(self) -> None:
        return

    @override
    def reduce(self) -> torch.Tensor:
        if not self.values:
            return torch.empty(0)

        return torch.cat(self.values)


class ReconstructionErrorMetric(Objective[Outputs, Targets]):
    """Metric computing reconstruction error statistics."""

    def __init__(self, denormalize: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.denormalize = denormalize
        return

    @override
    def calculate(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        recon_mu_unnorm = self.denormalize(outputs.recon_mu)
        dists = torch.sqrt(torch.sum((targets.x - recon_mu_unnorm) ** 2, dim=-1)) * 1000
        return {'error_mm': dists}

    @override
    def _compute(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, accumulator in self._aggregator.accumulators.items():
            distances = accumulator.reduce()
            if not distances.numel():
                continue

            out[f'{key}_mean'] = torch.mean(distances)
            out[f'{key}_std'] = torch.std(distances)
            out[f'{key}_median'] = torch.median(distances)
            out[f'{key}_>1'] = (distances > 1.0).float().mean()

        return out

    @classmethod
    @override
    def _get_accumulator_cls(cls) -> type[ReconstructionErrorCompiler]:
        return ReconstructionErrorCompiler


def get_nll(normalize: Callable[[torch.Tensor], torch.Tensor]) -> Loss[Outputs, Targets]:
    """Get the reconstruction NLL using a spherical Laplace decoder.

    The spherical Laplace distribution in 3D has density:
        p(x | mu, b) ∝ exp(-||x - mu||_2 / b)
    so NLL = 3*log(b) + (1/b) * ||x - mu||_2, summed over vertices.
    This is equivalent (up to constants) to MAE on per-vertex Euclidean norms,
    matching the evaluation metric.
    """

    def _nll(outputs: Outputs, targets: Targets) -> torch.Tensor:
        targets_normalized = normalize(targets.x)
        diff = targets_normalized - outputs.recon_mu
        per_vertex_norm = torch.norm(diff, dim=-1)
        log_b = outputs.recon_logvar.squeeze(-1)
        b = log_b.exp()
        per_vertex_nll = 3.0 * log_b + per_vertex_norm / b
        return per_vertex_nll.sum(dim=-1)

    return Loss(_nll, name='NLL')


def get_kld() -> LossWithMetrics[Outputs, Targets]:
    """Get the KL divergence loss (KLD)."""

    def _kld(outputs: Outputs, _: Targets) -> torch.Tensor:
        return -0.5 * (1 + outputs.logvar - outputs.mu.pow(2) - outputs.logvar.exp()).sum(1)

    return Loss(_kld, name='KLD')


def get_annealing(model: p.ModelProtocol[Any, Any], /) -> Loss[Outputs, Targets]:
    """Get the reverse annealing component.

    Args:
        model: the model whose epoch sets the weight.
    """
    annealing_period = Experiment.get_config().train.annealing_period

    def _annealing(outputs: Outputs, _: Targets) -> torch.Tensor:
        time_fraction = torch.tensor(model.epoch / annealing_period, device=outputs.recon_mu.device)
        time_fraction = torch.clamp(time_fraction, 0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(time_fraction * math.pi))

    return Loss(_annealing, name='Annealing')


def get_vae_loss(
    model: p.ModelProtocol[Any, Any],
    /,
    *,
    normalize: Callable[[torch.Tensor], torch.Tensor],
) -> LossWithMetrics[Outputs, Targets]:
    """Get VAE loss, composed with drytorch natural syntax.

    Args:
        model: the model the loss trains, whose epoch sets the annealing.
        normalize: callable to normalize target tensors.
    """
    cfg = Experiment.get_config()
    nll = get_nll(normalize)
    kld = get_kld()
    if cfg.train.annealing_period > 0:
        annealing = get_annealing(model)
        loss = nll + kld * annealing
    else:
        loss = nll + kld

    return loss


def get_reconstruction_metrics(
    denormalize: Callable[[torch.Tensor], torch.Tensor],
) -> Objective[Outputs, Targets]:
    """Get the reconstruction error metric using a denormalization callable."""
    return ReconstructionErrorMetric(denormalize)
