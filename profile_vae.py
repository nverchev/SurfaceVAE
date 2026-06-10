"""Script for profiling the Fixed Graph Variational AutoEncoder."""

import logging
import pathlib

import numpy as np
import torch

from src.config import AllConfig, Experiment, hydra_main, ModelOperators
from src.data import N_VERTICES, N_FACES
from src.data.structures import Input
from src.module import get_vae_module
from src.utils.profiling import save_parameters, profile_forward, profile_backward
from src.utils.sparse import scipy_sparse_to_pytorch_sparse
from src.data.operators import build_operator

logger = logging.getLogger(__name__)


def run_profiling(cfg: AllConfig) -> None:
    """Set up the loader and model, then run profiling."""
    device = cfg.user.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices, faces = _create_mock_mesh()
    mean_operator, mean_operator_adjoint = _get_mock_operators(
        vertices, faces, cfg.model.operator
    )
    mean_shape = torch.from_numpy(vertices).float()
    vae = get_vae_module(
        mean_shape=mean_shape,
        mean_operator=mean_operator,
        mean_operator_adjoint=mean_operator_adjoint,
    )
    profile_dir = _get_profiling_dir()
    inputs = _create_mock_inputs(
        cfg.train.batch_size, mean_operator, mean_operator_adjoint, device
    )
    save_parameters(vae, profile_dir)
    vae.train()
    vae.to(device)
    for _ in range(3):
        outputs = vae(inputs)
        loss = outputs.recon_mu.sum()
        loss.backward()
        vae.zero_grad()

    outputs = profile_forward(vae, inputs, device, profile_dir)
    loss = outputs.recon_mu.sum()
    profile_backward(loss, device, profile_dir)
    return


def _create_mock_inputs(
    batch_size: int,
    mean_operator: torch.Tensor,
    mean_operator_adjoint: torch.Tensor | None,
    device: torch.device,
) -> Input:
    """Create mock input namedtuple for the VAE model on the target device."""
    x = torch.randn(batch_size, N_VERTICES, 3, dtype=torch.float32, device=device)
    operator = mean_operator.to(device)
    if mean_operator_adjoint is not None:
        operator_adjoint = mean_operator_adjoint.to(device)
    else:
        operator_adjoint = torch.empty(0, device=device)

    return Input(x=x, operator=operator, operator_adjoint=operator_adjoint)


def _create_mock_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Create a mock mesh with deterministic connectivity."""
    vertices = np.random.randn(N_VERTICES, 3).astype(np.float32)
    faces = np.stack(
        [
            np.arange(N_FACES) % N_VERTICES,
            (np.arange(N_FACES) + 1) % N_VERTICES,
            (np.arange(N_FACES) + 2) % N_VERTICES,
        ],
        axis=1,
    )
    return vertices, faces


def _get_mock_operators(
    vertices: np.ndarray, faces: np.ndarray, operator_type: ModelOperators
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build mock operators as PyTorch sparse tensors."""
    op, op_adj = build_operator((vertices, faces), operator_type)
    if op is None:
        return torch.empty(0), torch.empty(0)

    mean_operator = scipy_sparse_to_pytorch_sparse(op)
    if op_adj is not None:
        mean_operator_adjoint = scipy_sparse_to_pytorch_sparse(op_adj)
    else:
        mean_operator_adjoint = None

    return mean_operator, mean_operator_adjoint


def _get_profiling_dir() -> pathlib.Path:
    exp = Experiment.get_current()
    if "@" in exp.run.id:
        day, time = exp.run.id.split("@")
        run_subdir = pathlib.Path(day) / time

    else:
        run_subdir = pathlib.Path(exp.run.id)

    profile_dir = exp.par_dir / "profile" / exp.name / run_subdir
    profile_dir.mkdir(exist_ok=True, parents=True)
    return profile_dir


def setup_and_profile(cfg: AllConfig) -> None:
    """Set up experiment and start VAE profiling."""
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )

    with exp.create_run(resume=True, record=False):
        run_profiling(cfg)

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Entry point for profiling."""
    setup_and_profile(cfg)
    return


if __name__ == "__main__":
    main()
