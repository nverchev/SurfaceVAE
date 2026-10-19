"""Events published during training and evaluation."""

from __future__ import annotations

import dataclasses

from collections.abc import Sequence
from typing import Any

import numpy as np

from numpy import typing as npt

from drytorch.core import log_events

__all__ = [
    'BONE',
    'MeshEvent',
]

BONE = np.array([0.89, 0.855, 0.788])


@dataclasses.dataclass(frozen=True)
class MeshEvent(log_events.Event):
    """Event logged when meshes are ready to be shown.

    The meshes share their topology, so they are drawn together and only the
    first carries the scalars.

    Attributes:
        vertices: Coordinates of every mesh to draw ``(n_vertices, 3)``.
        faces: Triangles the coordinates index ``(n_faces, 3)``.
        title: Name the drawing is filed under.
        folder: Directory to file it in, relative to the one the tracker owns.
        clim: Range the scalars span, or ``None`` for the range they take.
        cmap: Colour map the scalars are read through.
        color: Colour of the meshes that carry no scalars.
        scalars: Value of every vertex of the first mesh.
        show_edges: Whether the triangles are outlined.
        show_scalar_bar: Whether the range of the scalars is keyed.
        use_edl: Whether eye dome lighting sharpens the depth.
    """

    vertices: Sequence[npt.NDArray[Any]]
    faces: npt.NDArray[Any]
    title: str
    folder: str = ''
    clim: Sequence[float] | None = None
    cmap: str = 'viridis'
    color: str | Sequence[float] | npt.NDArray[Any] = dataclasses.field(default_factory=lambda: BONE)
    scalars: npt.NDArray[Any] | None = None
    show_edges: bool = False
    show_scalar_bar: bool = False
    use_edl: bool = False
