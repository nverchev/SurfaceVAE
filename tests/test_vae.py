"""Tests for the sample scaling of the VAE."""

import pytest
import torch

from torch import nn

from src.module.vae import BaseVAE


class _Scaler(BaseVAE):
    """Stand-in exposing only the buffers the scaling reads.

    ``BaseVAE.__init__`` builds the encoder and the decoder from the experiment
    configuration, which the scaling does not use, so this subclass initializes
    the module itself.
    """

    def __init__(self, mean_shape: torch.Tensor, *, use_mean_shape: bool) -> None:
        nn.Module.__init__(self)
        self.use_mean_shape = use_mean_shape
        self.register_buffer('mean_shape', mean_shape)
        self.register_buffer('mean_shape_center', mean_shape.mean(dim=0, keepdim=True))
        self.register_buffer('mean_shape_std', mean_shape.std())
        return


@pytest.fixture
def mean_shape() -> torch.Tensor:
    """Return a small mean shape standing in for the training average."""
    return torch.tensor([[0.0, 1.0, 2.0], [1.0, 3.0, -1.0], [-2.0, 0.5, 4.0]])


@pytest.mark.parametrize('use_mean_shape', [True, False])
def test_scaling_round_trip(mean_shape, use_mean_shape) -> None:
    """Denormalizing a normalized sample returns the original vertices."""
    scaler = _Scaler(mean_shape, use_mean_shape=use_mean_shape)
    sample = mean_shape + torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.5, -0.5], [1.0, 1.0, 1.0]])

    normalized = scaler.normalize_sample(sample)
    restored = scaler.denormalize_sample(normalized)

    torch.testing.assert_close(restored, sample)


def test_normalization_centres_the_mean_shape(mean_shape) -> None:
    """The mean shape normalizes to values centred on zero."""
    scaler = _Scaler(mean_shape, use_mean_shape=True)

    normalized = scaler.normalize_sample(mean_shape)

    torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
