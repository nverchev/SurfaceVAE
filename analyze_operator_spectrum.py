"""Analyze eigenvalue spectrum for any selected operator across dataset splits."""

import logging
import pathlib

from collections.abc import Sized
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import torch
from tqdm import tqdm

from src.config import AllConfig, Experiment, hydra_main
from src.config.options import ModelOperators
from src.data import build_operator, get_active_dataset
from src.data.operators import (
    build_lap_stiff,
    compute_mesh_geometry,
    compute_vertex_dual_areas,
)
from src.data.split import Partitions

logger = logging.getLogger(__name__)


def is_topology_only_operator(operator_type: ModelOperators) -> bool:
    """Check if the operator depends purely on graph topology (fixed connectivity)."""
    return operator_type in (
        ModelOperators.lap_graph_norm,
        ModelOperators.dirac_graph_norm,
    )


def compute_sample_spectrum(
    vertices: np.ndarray,
    faces: np.ndarray,
    operator_type: ModelOperators,
    num_evs: int,
) -> np.ndarray:
    """Compute sorted eigenvalues for a given mesh and operator type."""
    if operator_type == ModelOperators.lap_beltrami:
        L_cot = build_lap_stiff(vertices, faces)
        _, face_areas = compute_mesh_geometry(vertices, faces)
        vertex_dual_areas = compute_vertex_dual_areas(
            faces, face_areas, vertices.shape[0]
        )
        d_inv_sqrt = 1.0 / np.sqrt(vertex_dual_areas)
        D_inv_sqrt_mat = sparse.diags(d_inv_sqrt, 0)
        L_sym = D_inv_sqrt_mat @ L_cot @ D_inv_sqrt_mat
        evals, _ = spla.eigsh(
            L_sym.tocsc(),
            k=num_evs,
            sigma=-1e-5,
            which="LM",
        )
    elif operator_type in (
        ModelOperators.dirac,
        ModelOperators.dirac_stiff,
        ModelOperators.dirac_graph_norm,
    ):
        op_data = build_operator((vertices, faces), operator_type)
        op = op_data[0]
        if op is None:
            raise ValueError(f"Computed operator for {operator_type} is None.")

        S = op.T @ op
        evals, _ = spla.eigsh(
            S.tocsc(),
            k=num_evs,
            sigma=-1e-5,
            which="LM",
        )
    else:
        op_data = build_operator((vertices, faces), operator_type)
        op = op_data[0]
        if op is None:
            raise ValueError(f"Computed operator for {operator_type} is None.")

        evals, _ = spla.eigsh(
            op.tocsc(),
            k=num_evs,
            sigma=-1e-5,
            which="LM",
        )

    return np.sort(evals)


@torch.inference_mode()
def analyze_operator_spectrum() -> None:
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    operator_type = cfg.model.operator
    save_dir = cfg.user.path.version_dir / "spectrum" / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    dataset = get_active_dataset().get_split(partition)
    faces = dataset.faces
    num_evs = cfg_user.plot.num_eigenvectors
    if num_evs < 20:
        num_evs = 20

    n_samples = len(cast(Sized, dataset))
    all_eigenvalues = np.zeros((n_samples, num_evs))
    logger.info(
        f"Analyzing spectrum for operator '{operator_type.value}' on {n_samples} samples..."
    )
    if is_topology_only_operator(operator_type):
        dummy_vertices = dataset[0][0].x.numpy()
        single_spectrum = compute_sample_spectrum(
            dummy_vertices, faces, operator_type, num_evs
        )
        all_eigenvalues[:] = single_spectrum
        logger.info(
            f"Operator '{operator_type.value}' is topology-only; spectrum computed once."
        )

    else:
        for i in tqdm(
            range(n_samples), desc=f"Spectrum computation ({operator_type.value})"
        ):
            item = dataset[i]
            vertices = item[0].x.numpy()
            all_eigenvalues[i] = compute_sample_spectrum(
                vertices, faces, operator_type, num_evs
            )

    _save_csv_outputs(all_eigenvalues, operator_type, save_dir)
    _plot_and_save_distribution(all_eigenvalues, operator_type, save_dir)
    _print_summary(all_eigenvalues, operator_type)
    return


def _save_csv_outputs(
    eigenvalues: np.ndarray,
    operator_type: ModelOperators,
    save_dir: pathlib.Path,
) -> None:
    num_evs = eigenvalues.shape[1]
    raw_columns = [f"ev_{k + 1}" for k in range(num_evs)]
    df_raw = pd.DataFrame(eigenvalues, columns=raw_columns)
    df_raw.index.name = "sample_id"
    raw_csv_path = save_dir / f"spectrum_raw_{operator_type.value}.csv"
    df_raw.to_csv(raw_csv_path)

    summary_data = []
    for k in range(num_evs):
        vals = eigenvalues[:, k]
        summary_data.append(
            {
                "k": k + 1,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "p25": float(np.percentile(vals, 25)),
                "median": float(np.median(vals)),
                "p75": float(np.percentile(vals, 75)),
                "max": float(np.max(vals)),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = save_dir / f"spectrum_summary_{operator_type.value}.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved raw CSV to {raw_csv_path}")
    logger.info(f"Saved summary CSV to {summary_csv_path}")
    return


def _plot_and_save_distribution(
    eigenvalues: np.ndarray,
    operator_type: ModelOperators,
    save_dir: pathlib.Path,
) -> None:
    num_evs = eigenvalues.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = np.arange(1, num_evs + 1)
    axes[0].boxplot(
        [eigenvalues[:, k] for k in range(num_evs)],
        positions=positions,
        patch_artist=True,
    )
    axes[0].set_xlabel("Eigenvalue Index (k)")
    axes[0].set_ylabel("Eigenvalue Magnitude")
    axes[0].set_title(f"Distribution of First {num_evs} EVs ({operator_type.value})")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    means = np.mean(eigenvalues, axis=0)
    stds = np.std(eigenvalues, axis=0)
    axes[1].errorbar(positions, means, yerr=stds, fmt="-o", capsize=4, color="navy")
    axes[1].set_xlabel("Eigenvalue Index (k)")
    axes[1].set_ylabel("Mean Eigenvalue ± Std")
    axes[1].set_title(f"Spectrum Profile ({operator_type.value})")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = save_dir / f"spectrum_{operator_type.value}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved spectrum plot to {plot_path}")
    return


def _print_summary(
    eigenvalues: np.ndarray,
    operator_type: ModelOperators,
) -> None:
    num_evs = eigenvalues.shape[1]
    means = np.mean(eigenvalues, axis=0)
    stds = np.std(eigenvalues, axis=0)
    mins = np.min(eigenvalues, axis=0)
    maxs = np.max(eigenvalues, axis=0)
    logger.info("=" * 60)
    logger.info(f"Spectrum Summary Statistics ({operator_type.value})")
    logger.info("=" * 60)
    logger.info(f"{'Index (k)':<10}{'Mean':<14}{'Std':<14}{'Min':<14}{'Max':<14}")
    logger.info("-" * 60)
    for k in range(num_evs):
        logger.info(
            f"{k + 1:<10}{means[k]:<14.6f}{stds[k]:<14.6f}{mins[k]:<14.6f}{maxs[k]:<14.6f}"
        )

    logger.info("=" * 60)
    return


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(
        cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags
    )
    with exp.create_run(record=False):
        analyze_operator_spectrum()

    return


if __name__ == "__main__":
    main()
