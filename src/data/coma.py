"""COMA dataset class definitions and preprocessing logic."""

import abc
import enum
import pathlib

from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.config import Experiment
from src.config.options import Expressions, ModelOperators
from src.data import operators
from src.data.split import COMADatasetSplit, Partitions
from src.utils.meshes import read_ply
from src.utils.sparse import (
    get_h5_dataset,
    get_h5_group,
    is_sparse_matrix_group_valid,
    load_sparse_matrices_as_pytorch,
    load_sparse_matrix_as_pytorch,
    save_sparse_matrices_to_h5,
    save_sparse_matrix_to_h5,
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


class Singleton(abc.ABCMeta):
    """A metaclass that ensures only one instance of a class is created."""

    _instances: ClassVar[dict[type, object]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]

    @classmethod
    @abc.abstractmethod
    def get_faces(cls) -> np.ndarray:
        """Get the faces of the dataset."""


class BaseCOMAData(metaclass=Singleton):
    """Base class to load raw COMA data, compute partitions, and manage operator caching."""

    folder: ClassVar[str] = "COMA_data"

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.coma_dir = cfg.user.path.data_dir / self.folder
        self.on_the_fly = cfg.user.on_the_fly
        self.operator_type = cfg.model.operator
        self._setup_attributes(cfg)
        self.h5_path = self._get_h5_path()
        self.class_names = sorted([d.name for d in self.coma_dir.glob("FaceTalk_*")])
        if not self.h5_path.exists():
            self._preprocess_shapes_and_labels()

        with h5py.File(self.h5_path, "r") as f:
            self.faces = get_h5_dataset(f, H5Keys.FACES)[:]

        return

    @classmethod
    def get_faces(cls) -> np.ndarray:
        cfg = Experiment.get_config()
        coma_dir = cfg.user.path.data_dir / cls.folder
        ply_paths = sorted(coma_dir.glob("*/*/*.ply"))
        if not ply_paths:
            raise FileNotFoundError(f"No PLY files found in {coma_dir}")

        _, faces = read_ply(ply_paths[0])
        return faces

    def _setup_attributes(self, cfg: Any) -> None:
        return

    @abc.abstractmethod
    def _get_h5_path(self) -> pathlib.Path:
        """Get path to the HDF5 archive."""

    def get_split(self, partition: Partitions) -> COMADatasetSplit:
        if partition == Partitions.train_val:
            train_vertices, _, train_op, train_op_adj, train_lbls = self._load_split(
                Partitions.train,
            )
            val_vertices, _, val_op, val_op_adj, val_lbls = self._load_split(
                Partitions.val
            )
            vertices = np.concatenate([train_vertices, val_vertices], axis=0)
            labels = np.concatenate([train_lbls, val_lbls], axis=0)
            if train_op is not None and val_op is not None:
                if (
                    len(train_op) > 0
                    and len(val_op) > 0
                    and train_op[0] is train_op[-1]
                    and val_op[0] is val_op[-1]
                ):
                    operator = [train_op[0]] * (len(train_op) + len(val_op))
                else:
                    operator = train_op + val_op
            else:
                operator = None

            if train_op_adj is not None and val_op_adj is not None:
                if (
                    len(train_op_adj) > 0
                    and len(val_op_adj) > 0
                    and train_op_adj[0] is train_op_adj[-1]
                    and val_op_adj[0] is val_op_adj[-1]
                ):
                    adjoint_operator = [train_op_adj[0]] * (
                        len(train_op_adj) + len(val_op_adj)
                    )
                else:
                    adjoint_operator = train_op_adj + val_op_adj
            else:
                adjoint_operator = None

        else:
            vertices, _, operator, adjoint_operator, labels = self._load_split(
                partition
            )

        return COMADatasetSplit(
            shapes=vertices,
            faces=self.faces,
            operator=operator,
            adjoint_operator=adjoint_operator,
            labels=labels,
            class_names=self.class_names,
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

    def _get_sorted_ply_paths(self) -> list[pathlib.Path]:
        """Gather all .ply mesh paths in filesystem order."""
        return sorted(self.coma_dir.glob("*/*/*.ply"))

    def _get_subject_labels(self, paths: list[pathlib.Path]) -> np.ndarray:
        """Get subject labels from paths."""
        subject_dirs = sorted([d.name for d in self.coma_dir.glob("FaceTalk_*")])
        subject_to_id = {name: i for i, name in enumerate(subject_dirs)}
        labels = [subject_to_id.get(p.parent.parent.name, 0) for p in paths]
        return np.array(labels)

    def _load_shapes_from_paths(self, paths: list[pathlib.Path]) -> np.ndarray:
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
        np.ndarray,
    ]:
        """Load vertices, faces, operators, and adjoint operators from the HDF5 archive."""
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

        if not should_compute_on_the_fly and self.operator_type != ModelOperators.none:
            with h5py.File(self.h5_path, "a") as f:
                if not is_sparse_matrix_group_valid(f, group_name):
                    if group_name in f:
                        del f[group_name]

                    faces_tmp = get_h5_dataset(f, H5Keys.FACES)[:]
                    if operator_is_constant:
                        self._compute_constant_operator(faces_tmp, f)
                    else:
                        self._compute_group_operators(partition, faces_tmp, f)

        with h5py.File(self.h5_path, "r") as f:
            faces = get_h5_dataset(f, H5Keys.FACES)[:]
            vertices = get_h5_dataset(f, f"{partition.name}/{H5Keys.VERTICES}")[:]
            labels = get_h5_dataset(f, f"{partition.name}/{H5Keys.LABELS}")[:]
            if self.operator_type == ModelOperators.none:
                return vertices, faces, None, None, labels

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
            elif self.operator_type in (
                ModelOperators.dirac_norm,
                ModelOperators.dirac,
            ):
                adj_name = f"{partition.name}/{H5Keys.OPERATORS_ADJOINT}/{self.operator_type.name}"
                adjoint_operator = load_sparse_matrices_as_pytorch(
                    get_h5_group(f, adj_name)
                )
            else:
                adjoint_operator = None

        return vertices, faces, operator, adjoint_operator, labels

    def _preprocess_shapes_and_labels(self) -> None:
        ply_paths = self._get_sorted_ply_paths()
        _, faces = read_ply(ply_paths[0])
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

    @abc.abstractmethod
    def _split_paths(
        self, ply_paths: list[pathlib.Path]
    ) -> dict[Partitions, list[pathlib.Path]]:
        """Split paths into train, validation, and test sets."""


class COMAData(BaseCOMAData):
    """Standard COMA dataset loader."""

    def _get_h5_path(self) -> pathlib.Path:
        return self.coma_dir / "COMA_dataset.h5"

    def _split_paths(
        self, ply_paths: list[pathlib.Path]
    ) -> dict[Partitions, list[pathlib.Path]]:
        """Split paths into train, validation, and test sets."""
        train_paths: list[pathlib.Path] = []
        val_paths: list[pathlib.Path] = []
        test_paths: list[pathlib.Path] = []
        for idx, p in enumerate(ply_paths):
            mod = idx % 100
            if mod < 10:
                test_paths.append(p)

            elif mod < 20:
                val_paths.append(p)

            else:
                train_paths.append(p)

        return {
            Partitions.train: train_paths,
            Partitions.val: val_paths,
            Partitions.test: test_paths,
        }


class COMAExtrapolationData(BaseCOMAData):
    """COMA extrapolation dataset loader."""

    def _setup_attributes(self, cfg: Any) -> None:
        self.leave_out = cfg.data.dataset.leave_out
        return

    def _get_h5_path(self) -> pathlib.Path:
        return self.coma_dir / f"COMA_{self.leave_out}.h5"

    def _split_paths(
        self, ply_paths: list[pathlib.Path]
    ) -> dict[Partitions, list[pathlib.Path]]:
        """Split paths into train, validation, and test sets.

        A typical PLY path format is:
        `<data_dir>/COMA_data/<subject_id>/<expression>/<frame_name>.ply`
        - parts[-1] is the frame name (e.g., `lips_up.000001.ply`)
        - parts[-2] is the expression name (e.g., `lips_up`)
        - parts[-3] is the subject identity (e.g., `FaceTalk_170809_00138_TA`)
        """
        train_paths: list[pathlib.Path] = []
        val_paths: list[pathlib.Path] = []
        test_paths: list[pathlib.Path] = []
        val_expression = next(exp.name for exp in Expressions if exp != self.leave_out)
        for p in ply_paths:
            if len(p.parts) >= 2 and p.parts[-2] == self.leave_out:
                test_paths.append(p)
            elif len(p.parts) >= 2 and p.parts[-2] == val_expression:
                val_paths.append(p)
            else:
                train_paths.append(p)

        return {
            Partitions.train: train_paths,
            Partitions.val: val_paths,
            Partitions.test: test_paths,
        }
