"""Sparse matrix utilities."""

import enum

import h5py
import numpy as np
import scipy.sparse as sparse
import torch

torch.sparse.check_sparse_tensor_invariants.disable()


class SparseKeys(enum.StrEnum):
    """HDF5 sparse matrix keys."""

    ROW = "row"
    COL = "col"
    DATA = "data"
    SHAPE = "shape"


def scipy_sparse_to_pytorch_sparse(
    sparse_matrix: sparse.coo_matrix,
) -> torch.Tensor:
    """Convert scipy sparse matrix to PyTorch sparse COO tensor."""
    coo = sparse_matrix
    values = coo.data
    indices_numpy = np.vstack((coo.row, coo.col))
    indices_tensor = torch.LongTensor(indices_numpy)
    values_tensor = torch.FloatTensor(values)
    assert coo.shape is not None
    return torch.sparse_coo_tensor(indices_tensor, values_tensor, torch.Size(coo.shape))


def torch_block_diag_repeat(mat: torch.Tensor, B: int) -> torch.Tensor:
    if B == 1:
        return mat

    mat = mat.coalesce()
    indices = mat.indices()
    values = mat.values()
    shape = mat.size()
    new_values = values.repeat(B)
    row_offset = torch.arange(B, device=mat.device) * shape[0]
    col_offset = torch.arange(B, device=mat.device) * shape[1]
    offset_row = row_offset.unsqueeze(1).repeat(1, indices.size(1))
    offset_col = col_offset.unsqueeze(1).repeat(1, indices.size(1))
    offsets = torch.stack([offset_row, offset_col], dim=0)
    new_indices = indices.unsqueeze(1).repeat(1, B, 1) + offsets
    new_indices = new_indices.reshape(2, -1)
    new_size = (B * shape[0], B * shape[1])
    return torch.sparse_coo_tensor(new_indices, new_values, new_size)


def torch_sparse_block_diag(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.empty(0)

    if any(t.numel() == 0 for t in tensors):
        return torch.empty(0)

    coalesced_tensors = [t.coalesce() for t in tensors]
    all_indices = []
    all_values = []
    current_row_offset = 0
    current_col_offset = 0
    for t in coalesced_tensors:
        indices = t.indices()
        values = t.values()
        r, c = t.size()
        shifted_indices = indices.clone()
        shifted_indices[0] += current_row_offset
        shifted_indices[1] += current_col_offset
        all_indices.append(shifted_indices)
        all_values.append(values)
        current_row_offset += r
        current_col_offset += c

    new_indices = torch.cat(all_indices, dim=1)
    new_values = torch.cat(all_values, dim=0)
    new_size = (current_row_offset, current_col_offset)
    return torch.sparse_coo_tensor(new_indices, new_values, new_size)


def save_sparse_matrix_to_h5(
    group: h5py.Group,
    op: sparse.coo_matrix,
) -> None:
    """Save a single sparse COO matrix to an HDF5 group."""
    group.create_dataset(SparseKeys.ROW, data=op.row)
    group.create_dataset(SparseKeys.COL, data=op.col)
    group.create_dataset(SparseKeys.DATA, data=op.data)
    group.create_dataset(SparseKeys.SHAPE, data=np.array(op.shape))
    return


def save_sparse_matrices_to_h5(
    group: h5py.Group,
    op_list: list[sparse.coo_matrix],
) -> None:
    """Save a list of sparse COO matrices (with identical sparsity pattern) to an HDF5 group."""
    group.create_dataset(SparseKeys.ROW, data=op_list[0].row)
    group.create_dataset(SparseKeys.COL, data=op_list[0].col)
    n_matrices = len(op_list)
    nnz = len(op_list[0].data)
    ds = group.create_dataset(
        SparseKeys.DATA, shape=(n_matrices, nnz), dtype=np.float32
    )
    for idx, o in enumerate(op_list):
        ds[idx] = o.data

    group.create_dataset(SparseKeys.SHAPE, data=np.array(op_list[0].shape))
    return


def get_h5_dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    ds = group[key]
    assert isinstance(ds, h5py.Dataset)
    return ds


def get_h5_group(group: h5py.Group, key: str) -> h5py.Group:
    grp = group[key]
    assert isinstance(grp, h5py.Group)
    return grp


def load_sparse_matrix_from_h5(group: h5py.Group) -> sparse.coo_matrix:
    """Load a single sparse COO matrix from an HDF5 group."""
    row = get_h5_dataset(group, SparseKeys.ROW)[:]
    col = get_h5_dataset(group, SparseKeys.COL)[:]
    data = get_h5_dataset(group, SparseKeys.DATA)[:]
    shape = tuple(get_h5_dataset(group, SparseKeys.SHAPE)[:])
    return sparse.coo_matrix((data, (row, col)), shape=shape)


def load_sparse_matrices_from_h5(group: h5py.Group) -> list[sparse.coo_matrix]:
    """Load a list of sparse COO matrices from an HDF5 group."""
    row = get_h5_dataset(group, SparseKeys.ROW)[:]
    col = get_h5_dataset(group, SparseKeys.COL)[:]
    data = get_h5_dataset(group, SparseKeys.DATA)[:]
    shape = tuple(get_h5_dataset(group, SparseKeys.SHAPE)[:])
    return [sparse.coo_matrix((d, (row, col)), shape=shape) for d in data]


def load_sparse_matrix_as_pytorch(group: h5py.Group) -> torch.Tensor:
    """Load a single sparse COO matrix directly as a PyTorch sparse COO tensor."""
    row = get_h5_dataset(group, SparseKeys.ROW)[:]
    col = get_h5_dataset(group, SparseKeys.COL)[:]
    data = get_h5_dataset(group, SparseKeys.DATA)[:]
    shape = tuple(get_h5_dataset(group, SparseKeys.SHAPE)[:])
    indices = torch.stack(
        [torch.from_numpy(row).long(), torch.from_numpy(col).long()], dim=0
    )
    values = torch.from_numpy(data).float()
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))


def load_sparse_matrices_as_pytorch(group: h5py.Group) -> list[torch.Tensor]:
    """Load a list of sparse COO matrices directly as a list of PyTorch sparse COO tensors."""
    row = get_h5_dataset(group, SparseKeys.ROW)[:]
    col = get_h5_dataset(group, SparseKeys.COL)[:]
    data = get_h5_dataset(group, SparseKeys.DATA)[:]
    shape = tuple(get_h5_dataset(group, SparseKeys.SHAPE)[:])
    indices = torch.stack(
        [torch.from_numpy(row).long(), torch.from_numpy(col).long()], dim=0
    )
    values = torch.from_numpy(data).float()
    return [torch.sparse_coo_tensor(indices, val, torch.Size(shape)) for val in values]


def is_sparse_matrix_group_valid(f: h5py.File, group_name: str) -> bool:
    """Return True only if the group exists and contains the expected datasets."""
    if group_name not in f:
        return False

    group = f[group_name]
    if not isinstance(group, h5py.Group):
        return False

    return all(key.value in group for key in SparseKeys)
