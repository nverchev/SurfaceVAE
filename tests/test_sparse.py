"""Tests for the sparse conversion helpers."""

import numpy as np
import torch

from scipy import sparse

from src.utils.sparse import scipy_sparse_to_pytorch_sparse


def test_conversion_keeps_the_entries() -> None:
    """The converted tensor holds the same dense matrix as the scipy one."""
    matrix = sparse.coo_matrix(
        (np.array([1.5, -2.0, 3.25]), (np.array([0, 2, 2]), np.array([1, 0, 3]))),
        shape=(3, 4),
    )

    tensor = scipy_sparse_to_pytorch_sparse(matrix)

    assert tensor.shape == (3, 4)
    torch.testing.assert_close(tensor.to_dense(), torch.tensor(matrix.toarray(), dtype=torch.float32))
