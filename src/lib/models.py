"""DRYTorch Model subtypes and VAE wrapper module."""

from typing import Any

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import Model, EMAModel

import torch

from src.config import Experiment
from src.data.structures import Inputs, Outputs
from src.module.vaes import BaseVAE, get_vae_module
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
    load_trained_state(model, epoch=cfg.user.load or -1)
    return model.averaged_module


def load_trained_state(model: Model[Any, Any], /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.user.path.version_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return
