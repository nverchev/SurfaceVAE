"""DRYTorch Model subtypes and VAE wrapper module."""

from drytorch.core import protocols as p
from drytorch.lib.models import Model, EMAModel

import torch

from src.config import Experiment
from src.data.structures import Input, Output
from src.module.vae import BaseVAE, get_vae_module
from src.data import get_active_dataset, get_dummy_operator, N_VERTICES
from src.utils.sparse import scipy_sparse_to_pytorch_sparse


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


def load_extract_vae_module() -> BaseVAE:
    """Load and extract the VAE module from the EMA model."""
    cfg = Experiment.get_config()
    faces = get_active_dataset().get_faces()
    op, op_adj = get_dummy_operator(faces, N_VERTICES, cfg.model.operator)
    mean_operator = scipy_sparse_to_pytorch_sparse(op) if op is not None else None
    mean_operator_adjoint = (
        scipy_sparse_to_pytorch_sparse(op_adj) if op_adj is not None else None
    )
    mean_shape = torch.zeros(N_VERTICES, 3)
    vae = get_vae_module(
        mean_shape=mean_shape,
        mean_operator=mean_operator,
        mean_operator_adjoint=mean_operator_adjoint,
    ).eval()
    model = EMAModelEpoch(vae, name=f"VAE_{cfg.model.operator}", device=cfg.user.device)
    epoch = cfg.user.load_checkpoint or -1  # changes default to load last epoch
    model.load_state(epoch)
    ema_vae = model.averaged_module
    assert isinstance(ema_vae, BaseVAE)
    return ema_vae
