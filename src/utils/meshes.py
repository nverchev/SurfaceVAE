"""Module for loading and processing mesh files."""

import trimesh
import numpy as np


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY file using trimesh."""
    mesh = trimesh.load_mesh(path, process=False)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
