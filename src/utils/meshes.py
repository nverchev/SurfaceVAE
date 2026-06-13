"""Module for loading and processing mesh files."""

import pathlib

import trimesh
import numpy as np


def read_ply(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY file using trimesh."""
    mesh = trimesh.load_mesh(path, process=False)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
