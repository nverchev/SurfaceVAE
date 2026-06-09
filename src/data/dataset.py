"""COMA Dataset implementation and loader class."""

import enum
from typing import ClassVar
import pathlib

import glob
import h5py

import numpy as np
from tqdm import tqdm

import torch

from src.config import Experiment, ModelOperators
from src.data import operators
from src.data.split import (
    COMADatasetSplit,
    Partitions,
)
from src.utils.meshes import read_ply
from src.utils.sparse import (
    save_sparse_matrix_to_h5,
    save_sparse_matrices_to_h5,
    load_sparse_matrix_as_pytorch,
    load_sparse_matrices_as_pytorch,
    is_sparse_matrix_group_valid,
    get_h5_dataset,
    get_h5_group,
)


N_VERTICES = 5023
N_FACES = 9976


class H5Keys(enum.StrEnum):
    """HDF5 dataset keys and paths."""

    FACES = "faces"
    VERTICES = "vertices"
    LABELS = "labels"
    OPERATORS = "operators"
    OPERATORS_ADJOINT = "operators_adjoint"


class Singleton(type):
    """A metaclass that ensures only one instance of a class is created."""

    _instances: ClassVar[dict[type, object]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class COMAData(metaclass=Singleton):
    """Class to load raw COMA data, compute partitions, and manage operator caching."""

    folder: ClassVar[str] = "COMA_data"

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.coma_dir = cfg.user.path.data_dir / self.folder
        self.on_the_fly = cfg.user.on_the_fly
        self.h5_path = self.coma_dir / "COMA_dataset.h5"
        self.operator_type = cfg.model.operator
        self.vertices, self.faces, _, _ = self._load_split(Partitions.train)
        return

    def get_split(self, partition: Partitions) -> COMADatasetSplit:
        if partition == Partitions.train_val:
            train_vertices, _, train_op, train_op_adj = self._load_split(
                Partitions.train,
            )
            val_vertices, _, val_op, val_op_adj = self._load_split(Partitions.val)
            vertices = np.concatenate([train_vertices, val_vertices], axis=0)
            if train_op is not None and val_op is not None:
                operator = train_op + val_op
            else:
                operator = None

            if train_op_adj is not None and val_op_adj is not None:
                adjoint_operator = train_op_adj + val_op_adj
            else:
                adjoint_operator = None
        else:
            vertices, _, operator, adjoint_operator = self._load_split(partition)

        return COMADatasetSplit(
            shapes=vertices,
            faces=self.faces,
            operator=operator,
            adjoint_operator=adjoint_operator,
        )

    def _compute_constant_operator(self, faces: np.ndarray, f: h5py.File) -> None:
        train_shapes = get_h5_dataset(f, f"{Partitions.train.name}/{H5Keys.VERTICES}")[
            :
        ]
        op, _ = operators.build_operator((train_shapes[0], faces), self.operator_type)
        op_grp = f.create_group(self.operator_type.name)
        if op is not None:
            save_sparse_matrix_to_h5(op_grp, op)

        return

    def _compute_group_operators(
        self, partition: Partitions, faces: np.ndarray, f: h5py.File
    ) -> None:
        s_data = get_h5_dataset(f, f"{partition.name}/{H5Keys.VERTICES}")[:]
        op_list = []
        op_adj_list = []
        for shape in tqdm(
            s_data, desc=f"Computing {self.operator_type.name} for {partition.name}"
        ):
            op, op_adj = operators.build_operator((shape, faces), self.operator_type)
            if op is not None:
                op_list.append(op)
            
            if op_adj is not None:
                op_adj_list.append(op_adj)

        op_grp = f.create_group(
            f"{partition.name}/{H5Keys.OPERATORS}/{self.operator_type.name}"
        )
        save_sparse_matrices_to_h5(op_grp, op_list)
        if op_adj_list:
            op_adj_grp = f.create_group(
                f"{partition.name}/{H5Keys.OPERATORS_ADJOINT}/{self.operator_type.name}"
            )
            save_sparse_matrices_to_h5(op_adj_grp, op_adj_list)

        return

    def _get_sorted_ply_paths(self) -> list[str]:
        """Gather all .ply mesh paths in filesystem order."""
        return glob.glob(str(self.coma_dir / "*" / "*" / "*.ply"))

    def _get_subject_labels(self, paths: list[str]) -> np.ndarray:
        """Get subject labels from paths."""
        subject_dirs = sorted(
            [pathlib.Path(d).name for d in glob.glob(str(self.coma_dir / "FaceTalk_*"))]
        )
        subject_to_id = {name: i for i, name in enumerate(subject_dirs)}
        labels = [
            subject_to_id.get(pathlib.Path(p).parent.parent.name, 0) for p in paths
        ]
        return np.array(labels)

    def _load_shapes_from_paths(self, paths: list[str]) -> np.ndarray:
        """Load vertices from the given list of .ply paths."""
        shapes_list = []
        for path in tqdm(paths, desc="Loading PLY files"):
            vertices, _ = read_ply(path)
            shapes_list.append(vertices.astype(np.float32))

        return np.array(shapes_list)

    def _load_split(
        self, partition: Partitions
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        """Load vertices, faces, operators, and adjoint operators from the HDF5 archive."""
        if not self.h5_path.exists():
            self._preprocess_shapes_and_labels()

        with h5py.File(self.h5_path, "a") as f:
            faces = get_h5_dataset(f, H5Keys.FACES)[:]
            vertices = get_h5_dataset(f, f"{partition.name}/{H5Keys.VERTICES}")[:]
            if self.operator_type == ModelOperators.none:
                return vertices, faces, None, None

            operator_is_constant = self.operator_type in (
                ModelOperators.lap_graph_norm,
                ModelOperators.dirac_graph_norm,
            )
            group_name = (
                self.operator_type.name
                if operator_is_constant
                else f"{partition.name}/{H5Keys.OPERATORS}/{self.operator_type.name}"
            )
            should_compute_on_the_fly = not operator_is_constant and self.on_the_fly
            if not should_compute_on_the_fly:
                if not is_sparse_matrix_group_valid(f, group_name):
                    if group_name in f:
                        del f[group_name]

                    if operator_is_constant:
                        self._compute_constant_operator(faces, f)
                    else:
                        self._compute_group_operators(partition, faces, f)

            if should_compute_on_the_fly:
                operator = None
            elif operator_is_constant:
                op_const = load_sparse_matrix_as_pytorch(get_h5_group(f, group_name))
                operator = [op_const] * vertices.shape[0]
            else:
                operator = load_sparse_matrices_as_pytorch(get_h5_group(f, group_name))

            if should_compute_on_the_fly:
                adjoint_operator = None
            elif self.operator_type == ModelOperators.dirac_graph_norm:
                adjoint_operator = [op_const.T] * vertices.shape[0]
            elif self.operator_type == ModelOperators.dirac_norm:
                adj_name = f"{partition.name}/{H5Keys.OPERATORS_ADJOINT}/{self.operator_type.name}"
                adjoint_operator = load_sparse_matrices_as_pytorch(
                    get_h5_group(f, adj_name)
                )
            else:
                adjoint_operator = None

        return vertices, faces, operator, adjoint_operator

    def _preprocess_shapes_and_labels(self) -> None:
        ply_paths = self._get_sorted_ply_paths()
        _, faces = read_ply(ply_paths[0])  # All shapes have same topology
        split_paths_dict = self._split_paths(ply_paths)
        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset(H5Keys.FACES, data=faces)
            for dataset, paths in split_paths_dict.items():
                shapes = self._load_shapes_from_paths(paths)
                labels = self._get_subject_labels(paths)
                grp = f.create_group(dataset.name)
                grp.create_dataset(H5Keys.VERTICES, data=shapes)
                grp.create_dataset(H5Keys.LABELS, data=labels)

        return

    def _split_paths(self, ply_paths: list[str]) -> dict[Partitions, list[str]]:
        """Split paths into train, validation, and test sets."""
        indices = np.arange(len(ply_paths))
        test_mask = (indices % 100) < 10
        test_paths = [ply_paths[i] for i in indices[test_mask]]
        train_raw_paths = [ply_paths[i] for i in indices[~test_mask]]
        return {
            Partitions.train: train_raw_paths[:-100],
            Partitions.val: train_raw_paths[-100:],
            Partitions.test: test_paths,
        }


def _get_splits() -> tuple[COMADatasetSplit, COMADatasetSplit]:
    """Get training and evaluation dataloaders."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    train_dataset = COMAData().get_split(train_partition)
    eval_dataset = COMAData().get_split(eval_partition)
    return train_dataset, eval_dataset


def get_splits() -> tuple[COMADatasetSplit, COMADatasetSplit]:
    """Get training and evaluation dataloaders in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    splits: tuple[COMADatasetSplit, COMADatasetSplit] | None = None
    if cfg.user.n_subprocesses:
        import torch.distributed as dist
        rank = dist.get_rank()
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                splits = _get_splits()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])
    else:
        splits = _get_splits()

    if splits is None:
        raise RuntimeError("Splits could not be created.")

    return splits

