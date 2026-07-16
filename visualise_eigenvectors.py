"""Visualize eigenvectors of Laplacian, Dirac, or PCA for PointNet baseline."""

import logging
import random

from collections.abc import Sized

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.config.options import ModelOperators
from src.data import get_active_dataset, build_operator
from src.data.operators import (
    build_lap_stiff,
    compute_mesh_geometry,
    compute_vertex_dual_areas,
)
from src.data.split import Partitions
from src.utils.visualization import render_mesh

logger = logging.getLogger(__name__)


def extract_valid_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    num_evs: int,
) -> np.ndarray:
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    valid_mask = eigenvalues > 1e-6
    valid_vecs = eigenvectors[:, valid_mask]
    return valid_vecs[:, :num_evs]


def vec3_to_rgb(vecs: np.ndarray) -> np.ndarray:
    """Map (N, 3) signed vectors to (N, 3) uint8 RGB, with zero mapped to white.

    Uses a subtractive model: +x=red, -x=cyan, +y=green, -y=magenta, +z=blue, -z=yellow.
    """
    abs_max = np.percentile(np.abs(vecs), 95)
    if abs_max < 1e-8:
        return np.full((vecs.shape[0], 3), 255, dtype=np.uint8)

    v = (vecs / abs_max).clip(-1.0, 1.0)
    r = (
        1.0
        - np.maximum(0.0, v[:, 1])
        - np.maximum(0.0, v[:, 2])
        - np.maximum(0.0, -v[:, 0])
    )
    g = (
        1.0
        - np.maximum(0.0, v[:, 0])
        - np.maximum(0.0, v[:, 2])
        - np.maximum(0.0, -v[:, 1])
    )
    b = (
        1.0
        - np.maximum(0.0, v[:, 0])
        - np.maximum(0.0, v[:, 1])
        - np.maximum(0.0, -v[:, 2])
    )
    rgb = np.stack([r, g, b], axis=1).clip(0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


@torch.inference_mode()
def visualise_eigenvectors() -> None:
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / "images" / cfg.name / "eigenvectors"
    save_dir_base.mkdir(parents=True, exist_ok=True)
    if cfg_user.plot.use_train:
        dataset = get_active_dataset().get_split(Partitions.train)
    else:
        dataset = get_active_dataset().get_split(Partitions.test)

    faces = dataset.faces
    sample_indices = cfg_user.plot.sample_indices
    if not sample_indices:
        assert isinstance(dataset, Sized)
        sample_indices = [random.randint(0, len(dataset) - 1)]

    operator_type = cfg.model.operator
    num_evs = cfg_user.plot.num_eigenvectors
    for i in sample_indices:
        if i >= len(dataset):
            raise ValueError(
                f"Index {i} is too large for the selected dataset of length {len(dataset)}"
            )

        item = dataset[i]
        vertices = item[0].x.numpy()
        save_dir = save_dir_base / f"sample_{i}"
        save_dir.mkdir(parents=True, exist_ok=True)
        render_mesh(
            vertices,
            faces,
            title="original",
            interactive=interactive,
            save_dir=save_dir,
        )
        if operator_type == ModelOperators.lap_beltrami:
            L_cot = build_lap_stiff(vertices, faces)
            _, face_areas = compute_mesh_geometry(vertices, faces)
            vertex_dual_areas = compute_vertex_dual_areas(
                faces, face_areas, vertices.shape[0]
            )
            d_inv_sqrt = 1.0 / np.sqrt(vertex_dual_areas)
            D_inv_sqrt_mat = sparse.diags(d_inv_sqrt, 0)
            L_sym = D_inv_sqrt_mat @ L_cot @ D_inv_sqrt_mat
            eigenvalues, eigenvectors_sym = spla.eigsh(
                L_sym.tocsc(),
                k=num_evs + 25,
                sigma=1e-2,
                which="LM",
            )
            eigenvectors = D_inv_sqrt_mat @ eigenvectors_sym
        elif operator_type in (
            ModelOperators.dirac,
            ModelOperators.dirac_stiff,
            ModelOperators.dirac_graph_norm,
        ):
            op_data = build_operator((vertices, faces), operator_type)
            op = op_data[0]
            if op is None:
                raise ValueError("Computed operator is None.")

            S = op.T @ op
            eigenvalues, eigenvectors = spla.eigsh(
                S.tocsc(),
                k=num_evs + 25,
                sigma=1e-2,
                which="LM",
            )
        else:
            op_data = build_operator((vertices, faces), operator_type)
            op = op_data[0]
            if op is None:
                raise ValueError("Computed operator is None.")

            eigenvalues, eigenvectors = spla.eigsh(
                op.tocsc(),
                k=num_evs + 25,
                sigma=1e-2,
                which="LM",
            )

        valid_vecs = extract_valid_modes(eigenvalues, eigenvectors, num_evs)
        for col in range(valid_vecs.shape[1]):
            norm_val = np.linalg.norm(valid_vecs[:, col])
            if norm_val > 1e-8:
                valid_vecs[:, col] = valid_vecs[:, col] / norm_val

        for ev_idx in range(valid_vecs.shape[1]):
            ev = valid_vecs[:, ev_idx]
            if operator_type in (
                ModelOperators.dirac,
                ModelOperators.dirac_stiff,
                ModelOperators.dirac_graph_norm,
            ):
                ev_reshaped = ev.reshape(-1, 4)
                scalars = vec3_to_rgb(ev_reshaped[:, 1:])
                render_mesh(
                    vertices,
                    faces,
                    title=f"eigenvector_{ev_idx + 1}",
                    interactive=interactive,
                    save_dir=save_dir,
                    scalars=scalars,
                    show_scalar_bar=False,
                )

                scalars = ev_reshaped[:, 0]
                title = f"eigenvector_{ev_idx + 1}_real"
            else:
                scalars = ev
                title = f"eigenvector_{ev_idx + 1}"

            render_mesh(
                vertices,
                faces,
                title=title,
                interactive=interactive,
                save_dir=save_dir,
                scalars=scalars,
                cmap="coolwarm",
                show_scalar_bar=False,
            )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    with exp.create_run(resume=True):
        visualise_eigenvectors()

    return


if __name__ == "__main__":
    main()
