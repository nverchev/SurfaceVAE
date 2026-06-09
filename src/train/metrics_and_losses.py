"""Loss functions and metrics for VAE training."""

from __future__ import annotations

import math
from typing import Callable, Self, override
import torch

from drytorch.lib import aggregators
from drytorch.lib.objectives import (
    Loss,
    LossBase,
    Metric,
    Objective,
)
from src.config.experiment import Experiment
from src.data.structures import Output, Target

LOG_2PI = math.log(2 * math.pi)


class ReconstructionErrorCompiler(
    aggregators.AbstractAccumulator[torch.Tensor, dict[str, torch.Tensor]]
):
    """Accumulates distance tensors and reduces them to generative/error metrics."""

    def __init__(self, values: list[torch.Tensor]) -> None:
        self.values = values
        return

    @classmethod
    @override
    def from_value(cls, value: torch.Tensor) -> Self:
        return cls([value.detach().cpu()])

    @override
    def merge(self, other: Self) -> None:  # ty: ignore[invalid-method-override]
        self.values.extend(other.values)
        return

    @override
    def sync(self) -> None:
        return

    @override
    def reduce(self) -> dict[str, torch.Tensor]:
        if not self.values:
            return {}

        all_dists = torch.cat(self.values)
        return {
            "mean": torch.mean(all_dists),
            "std": torch.std(all_dists),
            "median": torch.median(all_dists),
            ">1": (all_dists > 1.0).float().mean(),
        }


class ReconstructionErrorAggregator(
    aggregators.AbstractAggregator[torch.Tensor, dict[str, torch.Tensor]]
):
    """Aggregator holding custom accumulators for reconstruction error."""

    accumulator_cls = ReconstructionErrorCompiler


class ReconstructionErrorMetric(Objective[Output, Target]):
    """Metric computing reconstruction error statistics."""

    def __init__(self, denormalize: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._aggregator: ReconstructionErrorAggregator = (
            ReconstructionErrorAggregator()
        )
        self.denormalize = denormalize
        return

    @override
    def calculate(self, outputs: Output, targets: Target) -> dict[str, torch.Tensor]:
        recon_mu_unnorm = self.denormalize(outputs.recon_mu)
        dists = torch.sqrt(torch.sum((targets.x - recon_mu_unnorm) ** 2, dim=-1)) * 1000
        return {"error_mm": dists}

    @override
    def _compute(self) -> dict[str, torch.Tensor]:
        raw = self._aggregator.reduce()
        out: dict[str, torch.Tensor] = {}
        for key, metrics in raw.items():
            for metric_name, value in metrics.items():
                out[f"{key}_{metric_name}"] = value

        return out

    @override
    def _get_aggregator(self) -> ReconstructionErrorAggregator:
        return self._aggregator


def log_normal_diag(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, sum_dim: int = 1
) -> torch.Tensor:
    """Compute the log probability of diagonal Gaussian."""
    return (-0.5 * (LOG_2PI + logvar + (z - mu).pow(2) / logvar.exp())).sum(sum_dim)


def get_nll(normalize: Callable[[torch.Tensor], torch.Tensor]) -> Loss[Output, Target]:
    """Get the reconstruction log-likelihood using a normalization callable."""
    cfg = Experiment.get_config()
    annealing_period = cfg.train.annealing_period

    def _nll(outputs: Output, targets: Target) -> torch.Tensor:
        targets_x_normalized = normalize(targets.x)
        x_flat = targets_x_normalized.view(targets.x.size(0), -1)
        recon_mu_flat = outputs.recon_mu.view(targets.x.size(0), -1)
        recon_logvar = outputs.recon_logvar.clamp(-6, -1)
        if outputs.model_epoch < annealing_period:
            recon_logvar = recon_logvar.detach()

        recon_logvar_flat = recon_logvar.expand_as(recon_mu_flat)
        return -log_normal_diag(x_flat, recon_mu_flat, recon_logvar_flat)

    return Loss(_nll, name="NLL")


def get_kld() -> LossBase[Output, Target]:
    """Get the KL divergence loss (KLD)."""

    def _kld(outputs: Output, _: Target) -> torch.Tensor:
        return -0.5 * (
            1 + outputs.logvar - outputs.mu.pow(2) - outputs.logvar.exp()
        ).sum(1)

    return Loss(_kld, name="KLD")


def get_pointwise_variance() -> Metric[Output, Target]:
    """Metric for average recon_var across batch."""

    def _pointwise_variance(outputs: Output, _: Target) -> torch.Tensor:
        return outputs.recon_logvar.exp()

    return Metric(_pointwise_variance, name="PVar")


def get_annealing() -> Loss[Output, Target]:
    """Get the reverse annealing component."""
    annealing_period = Experiment.get_config().train.annealing_period

    def _annealing(outputs: Output, _: Target) -> torch.Tensor:
        time_fraction = torch.tensor(
            outputs.model_epoch / annealing_period, device=outputs.recon_mu.device
        )
        time_fraction = torch.clamp(time_fraction, 0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(time_fraction * math.pi))

    return Loss(_annealing, name="Annealing")


def get_vae_loss(
    normalize: Callable[[torch.Tensor], torch.Tensor],
) -> LossBase[Output, Target]:
    """Get VAE loss, composed with drytorch natural syntax.

    Args:
        normalize: callable to normalize target tensors.
    """
    cfg = Experiment.get_config()
    nll = get_nll(normalize)
    kld = get_kld()
    if cfg.train.annealing_period > 0:
        annealing = get_annealing()
        loss = nll + kld * annealing
    else:
        loss = nll + kld

    loss.watch(get_pointwise_variance())
    return loss


def get_reconstruction_metrics(
    denormalize: Callable[[torch.Tensor], torch.Tensor],
) -> Objective[Output, Target]:
    """Get the reconstruction error metric using a denormalization callable."""
    return ReconstructionErrorMetric(denormalize)
