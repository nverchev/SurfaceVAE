"""Visualize eigenvectors of Laplacian, Dirac, or PCA for PointNet baseline."""

import logging
import random

from collections.abc import Sized

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import torch

from drytorch.lib import loading

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.train.events import MeshEvent
from src.config.options import ModelOperators
from src.data import get_databuilder, build_operator
from src.data.operators import (
    build_lap_stiff,
    compute_mesh_geometry,
    compute_vertex_dual_areas,
)
from src.train.loaders import collate_coma_batch
from src.data.split import Partitions

logger = logging.getLogger(__name__)


def extract_all_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    num_evs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract all sorted modes starting from the lowest eigenvalue."""
    idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[idx][:num_evs]
    sorted_vecs = eigenvectors[:, idx][:, :num_evs]
    return sorted_evals, sorted_vecs


def vec3_to_rgb(vecs: np.ndarray) -> np.ndarray:
    """Map (N, 3) signed vectors to (N, 3) uint8 RGB, with zero mapped to white.

    Uses a subtractive model: +x=red, -x=cyan, +y=green, -y=magenta, +z=blue, -z=yellow.
    """
    abs_max = np.percentile(np.abs(vecs), 95)
    if abs_max < 1e-8:
        return np.full((vecs.shape[0], 3), 255, dtype=np.uint8)

    v = (vecs / abs_max).clip(-1.0, 1.0)
    r = 1.0 - np.maximum(0.0, v[:, 1]) - np.maximum(0.0, v[:, 2]) - np.maximum(0.0, -v[:, 0])
    g = 1.0 - np.maximum(0.0, v[:, 0]) - np.maximum(0.0, v[:, 2]) - np.maximum(0.0, -v[:, 1])
    b = 1.0 - np.maximum(0.0, v[:, 0]) - np.maximum(0.0, v[:, 1]) - np.maximum(0.0, -v[:, 2])
    rgb = np.stack([r, g, b], axis=1).clip(0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


@torch.inference_mode()
def visualise_eigenvectors() -> None:
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if cfg_user.plot.use_train:
        split = get_databuilder().get_split(Partitions.train)
    else:
        split = get_databuilder().get_split(Partitions.test)

    faces = split.faces
    sample_indices = cfg_user.plot.sample_indices
    use_mean_shape = cfg_user.plot.use_mean_shape

    items_to_process = []
    if sample_indices:
        _, (inputs, _) = loading.take_from_dataset(split, indices=sample_indices, collate_fn=collate_coma_batch)
        for i, vertices in zip(sample_indices, inputs.x.unbind(), strict=True):
            items_to_process.append((f'sample_{i}', vertices.numpy()))

    elif use_mean_shape:
        items_to_process.append(('mean_shape', split.mean_shape.numpy()))
    else:
        assert isinstance(split, Sized)
        idx = random.randint(0, len(split) - 1)
        items_to_process.append((f'sample_{idx}', split[idx][0].x.numpy()))

    operator_type = cfg.model.operator
    num_evs = cfg_user.plot.num_eigenvectors
    for item_name, vertices in items_to_process:
        folder = f'eigenvectors/{item_name}'
        if operator_type == ModelOperators.LAP_BELTRAMI:
            L_cot = build_lap_stiff(vertices, faces)
            _, face_areas = compute_mesh_geometry(vertices, faces)
            vertex_dual_areas = compute_vertex_dual_areas(faces, face_areas, vertices.shape[0])
            d_inv_sqrt = 1.0 / np.sqrt(vertex_dual_areas)
            D_inv_sqrt_mat = sparse.diags(d_inv_sqrt, 0)
            L_sym = D_inv_sqrt_mat @ L_cot @ D_inv_sqrt_mat
            eigenvalues, eigenvectors_sym = spla.eigsh(
                L_sym.tocsc(),
                k=max(num_evs + 15, 25),
                sigma=1e-2,
                which='LM',
            )
            eigenvectors = D_inv_sqrt_mat @ eigenvectors_sym
        elif operator_type in (
            ModelOperators.DIRAC,
            ModelOperators.DIRAC_STIFF,
            ModelOperators.DIRAC_GRAPH_NORM,
        ):
            op_data = build_operator((vertices, faces), operator_type)
            op = op_data[0]
            if op is None:
                raise ValueError('Computed operator is None.')

            S = op.T @ op
            eigenvalues, eigenvectors = spla.eigsh(
                S.tocsc(),
                k=max(num_evs + 15, 25),
                sigma=1e-2,
                which='LM',
            )
        else:
            op_data = build_operator((vertices, faces), operator_type)
            op = op_data[0]
            if op is None:
                raise ValueError('Computed operator is None.')

            eigenvalues, eigenvectors = spla.eigsh(
                op.tocsc(),
                k=max(num_evs + 15, 25),
                sigma=1e-2,
                which='LM',
            )

        sorted_evals, sorted_vecs = extract_all_modes(eigenvalues, eigenvectors, num_evs)
        for col in range(sorted_vecs.shape[1]):
            norm_val = np.linalg.norm(sorted_vecs[:, col])
            if norm_val > 1e-8:
                sorted_vecs[:, col] = sorted_vecs[:, col] / norm_val

        for ev_idx in range(sorted_vecs.shape[1]):
            ev = sorted_vecs[:, ev_idx]
            ev_val = sorted_evals[ev_idx]
            eval_str = f'{ev_val:.6f}'
            if operator_type in (
                ModelOperators.DIRAC,
                ModelOperators.DIRAC_STIFF,
                ModelOperators.DIRAC_GRAPH_NORM,
            ):
                ev_reshaped = ev.reshape(-1, 4)
                scalars = vec3_to_rgb(ev_reshaped[:, 1:])
                MeshEvent(
                    vertices=[vertices],
                    faces=faces,
                    title=f'eigenvector_{ev_idx + 1}_eval_{eval_str}',
                    folder=folder,
                    scalars=scalars,
                )

                scalars = ev_reshaped[:, 0]
                title = f'eigenvector_{ev_idx + 1}_real_eval_{eval_str}'
            else:
                scalars = ev
                title = f'eigenvector_{ev_idx + 1}_eval_{eval_str}'

            MeshEvent(
                vertices=[vertices],
                faces=faces,
                title=title,
                folder=folder,
                scalars=scalars,
                cmap='coolwarm',
            )

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg):
        exp.trackers.subscribe(tracker)

    with exp.create_run(record=False):
        visualise_eigenvectors()

    return


if __name__ == '__main__':
    main()
