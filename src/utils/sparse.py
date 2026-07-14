"""Sparse matrix utilities."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import overload

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


class PreloadedSparseTensors(Sequence[torch.Tensor]):
    """Memory-efficient preloaded sequence of sparse tensors sharing indices."""

    def __init__(
        self, indices: torch.Tensor, values: torch.Tensor, shape: torch.Size
    ) -> None:
        self.indices = indices
        self.values = values
        self.shape = shape
        return

    def __len__(self) -> int:
        return self.values.shape[0]

    @overload
    def __getitem__(self, index: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> PreloadedSparseTensors: ...

    def __getitem__(self, index: int | slice) -> torch.Tensor | PreloadedSparseTensors:
        if isinstance(index, slice):
            sliced_values = self.values[index]
            return PreloadedSparseTensors(self.indices, sliced_values, self.shape)

        return torch.sparse_coo_tensor(self.indices, self.values[index], self.shape)


class CombinedSparseTensors(Sequence[torch.Tensor]):
    """Sequence that lazily concatenates multiple sequences of sparse tensors."""

    def __init__(self, sequences: list[Sequence[torch.Tensor]]) -> None:
        self.sequences = sequences
        self._lengths = [len(seq) for seq in sequences]
        self._total_len = sum(self._lengths)
        return

    def __len__(self) -> int:
        return self._total_len

    @overload
    def __getitem__(self, index: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[torch.Tensor]: ...

    def __getitem__(self, index: int | slice) -> torch.Tensor | Sequence[torch.Tensor]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._total_len)
            return [self[i] for i in range(start, stop, step)]

        if index < 0:
            index += self._total_len

        if index < 0 or index >= self._total_len:
            raise IndexError("Index out of range")

        curr_idx = index
        for seq, length in zip(self.sequences, self._lengths):
            if curr_idx < length:
                return seq[curr_idx]

            curr_idx -= length

        raise IndexError("Index out of range")


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


def get_h5_dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    ds = group[key]
    assert isinstance(ds, h5py.Dataset)
    return ds


def get_h5_group(group: h5py.Group, key: str) -> h5py.Group:
    grp = group[key]
    assert isinstance(grp, h5py.Group)
    return grp


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


def is_sparse_matrix_group_valid(f: h5py.File, group_name: str) -> bool:
    """Return True only if the group exists and contains the expected datasets."""
    if group_name not in f:
        return False

    group = f[group_name]
    if not isinstance(group, h5py.Group):
        return False

    return all(key.value in group for key in SparseKeys)


def load_sparse_matrices_as_pytorch_preloaded(
    group: h5py.Group,
) -> PreloadedSparseTensors:
    """Load a list of sparse COO matrices using a memory-efficient preloaded wrapper."""
    row = get_h5_dataset(group, SparseKeys.ROW)[:]
    col = get_h5_dataset(group, SparseKeys.COL)[:]
    data = get_h5_dataset(group, SparseKeys.DATA)[:]
    shape = tuple(get_h5_dataset(group, SparseKeys.SHAPE)[:])
    indices = torch.stack(
        [torch.from_numpy(row).long(), torch.from_numpy(col).long()], dim=0
    )
    values = torch.from_numpy(data).float()
    return PreloadedSparseTensors(indices, values, torch.Size(shape))


def create_incremental_sparse_dataset(
    group: h5py.Group,
    op: sparse.coo_matrix,
    n_matrices: int,
) -> h5py.Dataset:
    """Initialize datasets for saving sparse matrices incrementally and return the data dataset."""
    group.create_dataset(SparseKeys.ROW, data=op.row)
    group.create_dataset(SparseKeys.COL, data=op.col)
    group.create_dataset(SparseKeys.SHAPE, data=np.array(op.shape))
    nnz = len(op.data)
    ds = group.create_dataset(
        SparseKeys.DATA, shape=(n_matrices, nnz), dtype=np.float32
    )
    ds[0] = op.data
    return ds


def concat_operators(
    seq1: Sequence[torch.Tensor],
    seq2: Sequence[torch.Tensor],
) -> Sequence[torch.Tensor]:
    """Concatenate two operator sequences, supporting lists and custom preloaded sequences."""
    if isinstance(seq1, list) and isinstance(seq2, list):
        return seq1 + seq2

    sequences = []
    if isinstance(seq1, CombinedSparseTensors):
        sequences.extend(seq1.sequences)
    else:
        sequences.append(seq1)

    if isinstance(seq2, CombinedSparseTensors):
        sequences.extend(seq2.sequences)
    else:
        sequences.append(seq2)

    return CombinedSparseTensors(sequences)
