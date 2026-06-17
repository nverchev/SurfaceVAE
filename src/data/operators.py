"""Vectorized preprocessing functions for building fixed graph operators."""

from typing import cast

import numpy as np
import scipy.sparse as sparse

from src.config.options import ModelOperators

# ==============================================================================
# Basic Graph Connectivity
# ==============================================================================


def compute_directed_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract directed edge pairs (rows, cols) from face connectivity indices."""
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    rows = np.concatenate([v0, v1, v1, v2, v2, v0])
    cols = np.concatenate([v1, v0, v2, v1, v0, v2])
    return rows, cols


def compute_adjacency_matrix(faces: np.ndarray, n_vertices: int) -> sparse.coo_matrix:
    """Compute the binary adjacency matrix from face connectivity."""
    row, col = compute_directed_edges(faces)
    edges = np.unique(np.stack([row, col], axis=1), axis=0)
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(row), dtype=np.float32)
    return sparse.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices))


def compute_degree_inv_sqrt_matrix(W: sparse.coo_matrix) -> sparse.dia_matrix:
    """Compute the inverse square root diagonal matrix of the sum of weights."""
    degree_diagonal = np.array(W.sum(axis=1)).squeeze()
    degree_inv_sqrt = np.zeros_like(degree_diagonal)
    mask = degree_diagonal > 0
    degree_inv_sqrt[mask] = 1.0 / np.sqrt(degree_diagonal[mask])
    return sparse.diags(degree_inv_sqrt, 0)


# ==============================================================================
# Mesh Geometry
# ==============================================================================


def compute_face_edge_lengths(
    V: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three edge lengths for each face in the mesh."""
    v0 = F[:, 0]
    v1 = F[:, 1]
    v2 = F[:, 2]
    len_v0v1 = np.linalg.norm(V[v0] - V[v1], axis=-1)
    len_v1v2 = np.linalg.norm(V[v1] - V[v2], axis=-1)
    len_v2v0 = np.linalg.norm(V[v2] - V[v0], axis=-1)
    return len_v0v1, len_v1v2, len_v2v0


def compute_edge_distances(V: np.ndarray, F: np.ndarray) -> sparse.coo_matrix:
    """Compute the sparse distance matrix for edges in faces."""
    n_vertices = V.shape[0]
    len_v0v1, len_v1v2, len_v2v0 = compute_face_edge_lengths(V, F)
    rows, cols = compute_directed_edges(F)
    data = np.concatenate(
        [
            len_v0v1,
            len_v0v1,
            len_v1v2,
            len_v1v2,
            len_v2v0,
            len_v2v0,
        ]
    )
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))


def compute_edge_lengths_from_matrix(
    F: np.ndarray, edge_distances: sparse.coo_matrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the three edge lengths for each face from the sparse edge distance matrix."""
    v0 = F[:, 0]
    v1 = F[:, 1]
    v2 = F[:, 2]
    edge_distances_csr = edge_distances.tocsr()
    l_v0v1 = np.asarray(edge_distances_csr[v0, v1]).squeeze()
    l_v1v2 = np.asarray(edge_distances_csr[v1, v2]).squeeze()
    l_v2v0 = np.asarray(edge_distances_csr[v2, v0]).squeeze()
    return l_v0v1, l_v1v2, l_v2v0


def compute_heron_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute face areas using Heron's formula from edge lengths."""
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    return np.sqrt(np.clip(area_sq, 1e-10, None))


def compute_face_areas(F: np.ndarray, edge_distances: sparse.coo_matrix) -> np.ndarray:
    """Compute the area of each face from face vertex indices and edge lengths matrix."""
    l_v0v1, l_v1v2, l_v2v0 = compute_edge_lengths_from_matrix(F, edge_distances)
    return compute_heron_area(l_v0v1, l_v1v2, l_v2v0)


def compute_mesh_geometry(
    V: np.ndarray, F: np.ndarray
) -> tuple[sparse.coo_matrix, np.ndarray]:
    """Compute the sparse distance matrix and face areas for the given mesh."""
    edge_distances = compute_edge_distances(V, F)
    face_areas = compute_face_areas(F, edge_distances)
    return edge_distances, face_areas


# ==============================================================================
# Category 3: Laplace-Beltrami Math Helpers
# ==============================================================================


def compute_cotangent_formula(
    l_ij: np.ndarray, l_jk: np.ndarray, l_ki: np.ndarray, face_areas: np.ndarray
) -> np.ndarray:
    """Compute the cotangent weight for the edge ij opposite to the vertex k in a face."""
    return (-(l_ij**2) + l_jk**2 + l_ki**2) / (8 * face_areas)


def compute_vertex_dual_areas(
    faces: np.ndarray, face_areas: np.ndarray, n_vertices: int
) -> np.ndarray:
    """Sum face areas divided by 3 to vertices sharing each face."""
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    vertex_dual_areas = np.zeros(n_vertices)
    np.add.at(vertex_dual_areas, v0, face_areas / 3)
    np.add.at(vertex_dual_areas, v1, face_areas / 3)
    np.add.at(vertex_dual_areas, v2, face_areas / 3)
    return vertex_dual_areas


def compute_inverse_diagonal_matrix(
    faces: np.ndarray, face_areas: np.ndarray, n_vertices: int
) -> sparse.dia_matrix:
    """Compute the inverse diagonal dual area matrix from face areas."""
    vertex_dual_areas = compute_vertex_dual_areas(faces, face_areas, n_vertices)
    return sparse.diags(1.0 / vertex_dual_areas, 0)


def compute_cotangent_weights(
    F: np.ndarray, face_areas: np.ndarray, edge_distances: sparse.coo_matrix
) -> sparse.coo_matrix:
    """Compute cotangent weights matrix."""
    assert edge_distances.shape is not None
    n_vertices = edge_distances.shape[0]
    v0 = F[:, 0]
    v1 = F[:, 1]
    v2 = F[:, 2]
    l_v0v1, l_v1v2, l_v2v0 = compute_edge_lengths_from_matrix(F, edge_distances)
    cot_weight_v0v1 = compute_cotangent_formula(l_v0v1, l_v1v2, l_v2v0, face_areas)
    cot_weight_v1v2 = compute_cotangent_formula(l_v1v2, l_v2v0, l_v0v1, face_areas)
    cot_weight_v2v0 = compute_cotangent_formula(l_v2v0, l_v0v1, l_v1v2, face_areas)
    rows = np.concatenate([v0, v1, v1, v2, v2, v0])
    cols = np.concatenate([v1, v0, v2, v1, v0, v2])
    data = np.concatenate(
        [
            cot_weight_v0v1,
            cot_weight_v0v1,
            cot_weight_v1v2,
            cot_weight_v1v2,
            cot_weight_v2v0,
            cot_weight_v2v0,
        ]
    )
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    return W


def compute_laplacian(W: sparse.coo_matrix) -> sparse.coo_matrix:
    """Return the Laplacian of the weight matrix.

    Builds L = D - W directly from COO arrays without sparse arithmetic, so
    nnz is always len(W.data) + n regardless of whether any weights are zero.
    """
    assert W.shape is not None
    n = W.shape[0]
    D_diag = np.zeros(n)
    np.add.at(D_diag, W.row, W.data)
    rows = np.concatenate([W.row, np.arange(n)])
    cols = np.concatenate([W.col, np.arange(n)])
    data = np.concatenate([np.negative(W.data), D_diag])
    return sparse.coo_matrix((data, (rows, cols)), shape=W.shape)


# ==============================================================================
# Dirac Operator Mathematics
# ==============================================================================


def compute_quaternion_matrix(x: np.ndarray) -> np.ndarray:
    """Return the 4x4 real matrix representation of a quaternion."""
    a, b, c, d = x.tolist()
    return np.array(
        [
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a],
        ]
    )


def compute_quaternion_matrix_vectorized(q: np.ndarray) -> np.ndarray:
    """Build the 4x4 matrix representation of quaternions in a vectorized manner from [N, 4] vectors."""
    a = q[:, 0]
    b = q[:, 1]
    c = q[:, 2]
    d = q[:, 3]
    return np.stack(
        [
            np.stack([a, -b, -c, -d], axis=-1),
            np.stack([b, a, -d, c], axis=-1),
            np.stack([c, d, a, -b], axis=-1),
            np.stack([d, -c, b, a], axis=-1),
        ],
        axis=-2,
    )


def compute_corner_edge_vectors(
    V: np.ndarray, F: np.ndarray, corner: int
) -> np.ndarray:
    """For a given face corner (0, 1, 2), return the opposite edge vector (v_next1 - v_next2)."""
    v_next1 = F[:, (corner + 1) % 3]
    v_next2 = F[:, (corner + 2) % 3]
    return V[v_next1] - V[v_next2]


def compute_block_indices(
    row_block_indices: np.ndarray,
    col_block_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand block row/col indices to element-wise indices for 4x4 sub-matrices."""
    dx = np.arange(4)
    dy = np.arange(4)
    grid_x, grid_y = np.meshgrid(dx, dy, indexing="ij")
    rows = (4 * row_block_indices)[:, None, None] + grid_x[None, :, :]
    cols = (4 * col_block_indices)[:, None, None] + grid_y[None, :, :]
    return rows.ravel(), cols.ravel()


def compute_dirac_base(
    V: np.ndarray, F: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute the indices and quaternion matrices for Dirac operators."""
    n_faces = F.shape[0]
    f2v_rows = []
    f2v_cols = []
    quaternion_mats = []
    for corner in range(3):
        vertex_corner = F[:, corner]
        edge_vectors = compute_corner_edge_vectors(V, F, corner)
        q_edges = np.zeros((n_faces, 4))
        q_edges[:, 1:] = edge_vectors
        quaternion_matrices = -compute_quaternion_matrix_vectorized(q_edges)
        rows_f2v, cols_f2v = compute_block_indices(np.arange(n_faces), vertex_corner)
        f2v_rows.append(rows_f2v)
        f2v_cols.append(cols_f2v)
        quaternion_mats.append(quaternion_matrices)

    return f2v_rows, f2v_cols, quaternion_mats


def compute_graph_dirac(F: np.ndarray, n_vertices: int) -> sparse.coo_matrix:
    """Compute the unnormalized combinatorial graph Dirac operator from canonical equilateral embedding."""
    canonical_vertices = np.array(
        [
            [1.0 / np.sqrt(2), 0.0, 0.0],
            [0.0, 1.0 / np.sqrt(2), 0.0],
            [0.0, 0.0, 1.0 / np.sqrt(2)],
        ]
    )
    n_faces = F.shape[0]
    f2v_rows = []
    f2v_cols = []
    data = []
    for corner in range(3):
        vertex_corner = F[:, corner]
        q_edge = np.zeros(4)
        q_edge[1:] = (
            canonical_vertices[(corner + 1) % 3] - canonical_vertices[(corner + 2) % 3]
        )
        q_mat = -compute_quaternion_matrix(q_edge)
        rows_f2v, cols_f2v = compute_block_indices(np.arange(n_faces), vertex_corner)
        f2v_rows.append(rows_f2v)
        f2v_cols.append(cols_f2v)
        data.append(np.tile(q_mat, (n_faces, 1, 1)).ravel())

    return sparse.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(f2v_rows), np.concatenate(f2v_cols)),
        ),
        shape=(4 * n_faces, 4 * n_vertices),
    )


def compute_dirac(
    V: np.ndarray,
    F: np.ndarray,
    face_areas: np.ndarray,
    normalize: bool = True,
) -> tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """Compute the continuous Dirac operators (Di, DiA)."""
    n_vertices = V.shape[0]
    n_faces = F.shape[0]
    f2v_rows, f2v_cols, quaternion_mats = compute_dirac_base(V, F)
    vertex_dual_areas = compute_vertex_dual_areas(F, face_areas, n_vertices)
    f2v_data = []
    v2f_rows = []
    v2f_cols = []
    v2f_data = []
    for corner, q_mats in enumerate(quaternion_mats):
        vertex_corner = F[:, corner]
        if normalize:
            data_f2v = q_mats / (2 * face_areas[:, None, None])
        else:
            data_f2v = q_mats

        f2v_data.append(data_f2v.ravel())
        rows_v2f, cols_v2f = compute_block_indices(vertex_corner, np.arange(n_faces))
        if normalize:
            data_v2f = -q_mats / (2 * vertex_dual_areas[vertex_corner, None, None])
        else:
            data_v2f = -q_mats

        v2f_rows.append(rows_v2f)
        v2f_cols.append(cols_v2f)
        v2f_data.append(data_v2f.ravel())

    Di = sparse.coo_matrix(
        (
            np.concatenate(f2v_data),
            (np.concatenate(f2v_rows), np.concatenate(f2v_cols)),
        ),
        shape=(4 * n_faces, 4 * n_vertices),
    )
    DiA = sparse.coo_matrix(
        (
            np.concatenate(v2f_data),
            (np.concatenate(v2f_rows), np.concatenate(v2f_cols)),
        ),
        shape=(4 * n_vertices, 4 * n_faces),
    )
    return Di, DiA


# ==============================================================================
# Dummy Operator
# ==============================================================================


def buid_dummy_sparse_tensor(
    shape: tuple[int, int] | None, nnz: int
) -> sparse.coo_matrix | None:
    if shape is not None and nnz > 0:
        idx = np.arange(nnz, dtype=np.int64)
        rows = idx // shape[1]
        cols = idx % shape[1]
        data = np.zeros(nnz, dtype=np.float32)
        return sparse.coo_matrix((data, (rows, cols)), shape=shape)

    return None


# ==============================================================================
# Operator Builders
# ==============================================================================


def build_normalized_graph_laplacian(
    faces: np.ndarray, n_vertices: int
) -> sparse.coo_matrix:
    """Build the unweighted symmetric normalized graph Laplacian from face connectivity."""
    adjacency_matrix = compute_adjacency_matrix(faces, n_vertices)
    L_unnormalized = compute_laplacian(adjacency_matrix)
    degree_inv_sqrt_matrix = compute_degree_inv_sqrt_matrix(adjacency_matrix)
    return (degree_inv_sqrt_matrix * L_unnormalized * degree_inv_sqrt_matrix).tocoo()


def build_laplace_beltrami(V: np.ndarray, F: np.ndarray) -> sparse.coo_matrix:
    """Build the unnormalized cotangent-weighted Laplace-Beltrami operator."""
    edge_distances, face_areas = compute_mesh_geometry(V, F)
    W = compute_cotangent_weights(F, face_areas, edge_distances)
    return compute_laplacian(W)


def build_laplace_beltrami_normalized(
    V: np.ndarray, F: np.ndarray
) -> sparse.coo_matrix:
    """Build the normalized cotangent-weighted Laplace-Beltrami operator."""
    n_vertices = V.shape[0]
    edge_distances, face_areas = compute_mesh_geometry(V, F)
    W = compute_cotangent_weights(F, face_areas, edge_distances)
    L = compute_laplacian(W)
    vertex_dual_areas = compute_vertex_dual_areas(F, face_areas, n_vertices)
    normalized_data = L.data / vertex_dual_areas[L.row]
    return sparse.coo_matrix((normalized_data, (L.row, L.col)), shape=L.shape)


def build_dirac_graph_norm(faces: np.ndarray, n_vertices: int) -> sparse.coo_matrix:
    """Build the normalised combinatorial graph Dirac operator (D_tilde = D_comb @ D_deg^{-1/2})."""
    adjacency_matrix = compute_adjacency_matrix(faces, n_vertices)
    degree_inv_sqrt_matrix = compute_degree_inv_sqrt_matrix(adjacency_matrix)
    Di_comb = compute_graph_dirac(faces, n_vertices)
    D_deg_block = sparse.kron(degree_inv_sqrt_matrix, sparse.eye(4))
    return cast(sparse.coo_matrix, Di_comb @ D_deg_block).tocoo()


def build_dirac_norm(
    V: np.ndarray, F: np.ndarray
) -> tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """Build the continuous Dirac operators (Di, DiA)."""
    _, face_areas = compute_mesh_geometry(V, F)
    Di, DiA = compute_dirac(V, F, face_areas, normalize=True)
    return Di, DiA


def build_dirac(
    V: np.ndarray, F: np.ndarray
) -> tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """Build the continuous Dirac operators without area-normalization (Di, DiA)."""
    _, face_areas = compute_mesh_geometry(V, F)
    Di, DiA = compute_dirac(V, F, face_areas, normalize=False)
    return Di, DiA


# ==============================================================================


def build_operator(
    sample: tuple[np.ndarray, np.ndarray], operator: ModelOperators
) -> tuple[sparse.coo_matrix | None, sparse.coo_matrix | None]:
    """Process a single shape (vertices, faces) into operators in a vectorized manner."""
    vertices, faces = sample
    if operator == ModelOperators.none:
        return None, None

    if operator == ModelOperators.lap_graph_norm:
        return build_normalized_graph_laplacian(faces, vertices.shape[0]), None

    if operator == ModelOperators.dirac_graph_norm:
        op = build_dirac_graph_norm(faces, vertices.shape[0])
        return op, op.T.tocoo()

    if operator == ModelOperators.lap_beltrami:
        return build_laplace_beltrami(vertices, faces), None

    if operator == ModelOperators.lap_beltrami_norm:
        return build_laplace_beltrami_normalized(vertices, faces), None

    if operator == ModelOperators.dirac_norm:
        return build_dirac_norm(vertices, faces)

    if operator == ModelOperators.dirac:
        return build_dirac(vertices, faces)

    raise ValueError("Unknown operator")


def get_dummy_operator(
    faces: np.ndarray, n_vertices: int, operator_type: ModelOperators
) -> tuple[sparse.coo_matrix | None, sparse.coo_matrix | None]:
    if operator_type == ModelOperators.none:
        return None, None

    if operator_type in (
        ModelOperators.lap_graph_norm,
        ModelOperators.lap_beltrami,
        ModelOperators.lap_beltrami_norm,
    ):
        row, col = compute_directed_edges(faces)
        edges = np.unique(np.stack([row, col], axis=1), axis=0)
        E = len(edges)
        nnz = E + n_vertices
        op_shape = (n_vertices, n_vertices)
        op_nnz = nnz
        op_adj_shape = None
        op_adj_nnz = 0

    elif operator_type in (
        ModelOperators.dirac_graph_norm,
        ModelOperators.dirac_norm,
        ModelOperators.dirac,
    ):
        n_faces = faces.shape[0]
        nnz = 48 * n_faces
        op_shape = (4 * n_faces, 4 * n_vertices)
        op_nnz = nnz
        op_adj_shape = (4 * n_vertices, 4 * n_faces)
        op_adj_nnz = nnz

    else:
        raise ValueError("Unknown operator")

    mean_operator = buid_dummy_sparse_tensor(op_shape, op_nnz)
    mean_operator_adjoint = buid_dummy_sparse_tensor(op_adj_shape, op_adj_nnz)
    return mean_operator, mean_operator_adjoint
