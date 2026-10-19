"""DRYTorch Model subtypes and VAE wrapper module."""

from drytorch.lib.models import Model, EMAModel

import torch

from src.config import Experiment
from src.data.structures import Inputs, Outputs
from src.module.vae import BaseVAE, get_vae_module
from src.data import get_databuilder, get_dummy_operator, N_VERTICES
from src.utils.sparse import scipy_sparse_to_pytorch_sparse


class VAEModel(Model[Inputs, Outputs]):
    """Model wrapping the VAE.

    Attributes:
        module: the VAE.
    """

    module: BaseVAE


class VAEEMAModel(EMAModel[Inputs, Outputs], VAEModel):
    """VAE with exponential moving average weights.

    Attributes:
        module: the VAE.
        averaged_module: the VAE with the averaged weights.
    """

    module: BaseVAE
    averaged_module: BaseVAE


def load_extract_vae_module() -> BaseVAE:
    """Load and extract the VAE module from the EMA model."""
    cfg = Experiment.get_config()
    faces = get_databuilder().get_faces()
    op, op_adj = get_dummy_operator(faces, N_VERTICES, cfg.model.operator)
    mean_operator = scipy_sparse_to_pytorch_sparse(op) if op is not None else None
    mean_operator_adjoint = scipy_sparse_to_pytorch_sparse(op_adj) if op_adj is not None else None
    mean_shape = torch.zeros(N_VERTICES, 3)
    vae = get_vae_module(
        mean_shape=mean_shape,
        mean_operator=mean_operator,
        mean_operator_adjoint=mean_operator_adjoint,
    ).eval()
    model = VAEEMAModel(vae, name=f'VAE_{cfg.model.operator}', device=cfg.user.device)
    epoch = cfg.user.load or -1  # changes default to load last epoch
    model.load_state(epoch)
    return model.averaged_module
