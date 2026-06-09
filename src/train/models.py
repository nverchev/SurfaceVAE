"""DRYTorch Model subtypes and VAE wrapper module."""

from drytorch.core import protocols as p
from drytorch.lib.models import Model, EMAModel

from src.data.structures import Input, Output


class LogEpochMixin(p.ModelProtocol[Input, Output]):
    """Mixin to log the epoch to the outputs."""

    def __call__(self, inputs: Input) -> Output:
        out = super().__call__(inputs)
        out.model_epoch = self.epoch
        return out


class ModelEpoch(LogEpochMixin, Model[Input, Output]):
    """Model with epoch logging."""


class EMAModelEpoch(LogEpochMixin, EMAModel[Input, Output]):
    """EMA Model with epoch logging."""
