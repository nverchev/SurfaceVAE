"""Tests for the operator identities of the ``operators`` module."""

import numpy as np
import pytest

from scipy import sparse

from src.data.operators import (
    build_dirac,
    build_dirac_graph_norm,
    build_dirac_stiff,
    build_lap_stiff,
    build_laplace_beltrami,
    build_normalized_graph_laplacian,
)

TETRAHEDRON = (
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
)
OCTAHEDRON = (
    np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]),
    np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]),
)
MESHES = [TETRAHEDRON, OCTAHEDRON]


def _compose(adjoint: sparse.coo_matrix, dirac: sparse.coo_matrix) -> np.ndarray:
    """Return the dense composition of an operator with its adjoint."""
    return adjoint.toarray() @ dirac.toarray()


def _quaternion_real_part(product: np.ndarray) -> np.ndarray:
    """Return the scalar of each quaternion block of a 4n x 4n product."""
    blocks = product.reshape(product.shape[0] // 4, 4, product.shape[1] // 4, 4)
    scalars = blocks[:, 0, :, 0]
    for component in (1, 2, 3):
        np.testing.assert_allclose(blocks[:, component, :, component], scalars, atol=1e-10)

    return scalars


@pytest.mark.parametrize('vertices, faces', MESHES)
def test_dirac_squares_to_laplace_beltrami(vertices, faces) -> None:
    """Four times the Dirac pair product is the mass-normalised cotangent Laplacian."""
    dirac, adjoint = build_dirac(vertices, faces)

    product = _quaternion_real_part(_compose(adjoint, dirac))

    np.testing.assert_allclose(4 * product, build_laplace_beltrami(vertices, faces).toarray(), atol=1e-10)


@pytest.mark.parametrize('vertices, faces', MESHES)
def test_dirac_stiff_squares_to_stiffness(vertices, faces) -> None:
    """Four times the area-scaled Dirac pair product is the cotangent stiffness matrix."""
    dirac, adjoint = build_dirac_stiff(vertices, faces)

    product = _quaternion_real_part(_compose(adjoint, dirac))

    np.testing.assert_allclose(4 * product, build_lap_stiff(vertices, faces).toarray(), atol=1e-10)


@pytest.mark.parametrize('vertices, faces', MESHES)
def test_graph_dirac_squares_to_normalized_laplacian(vertices, faces) -> None:
    """The topology-only Dirac pair product is the normalised graph Laplacian."""
    n_vertices = vertices.shape[0]
    dirac, adjoint = build_dirac_graph_norm(faces, n_vertices)

    product = _quaternion_real_part(_compose(adjoint, dirac))

    np.testing.assert_allclose(product, build_normalized_graph_laplacian(faces, n_vertices).toarray(), atol=1e-10)


@pytest.mark.parametrize('vertices, faces', MESHES)
def test_dirac_maps_vertices_to_faces(vertices, faces) -> None:
    """The Dirac operator has one quaternion row per face and one column per vertex."""
    dirac, adjoint = build_dirac(vertices, faces)

    assert dirac.shape == (4 * faces.shape[0], 4 * vertices.shape[0])
    assert adjoint.shape == (4 * vertices.shape[0], 4 * faces.shape[0])
