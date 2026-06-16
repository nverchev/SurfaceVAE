"""Visualization utilities for meshes using PyVista."""

import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy import typing as npt

BLUE = np.array([0.3, 0.3, 0.9])
RED = np.array([0.9, 0.3, 0.3])
GREEN = np.array([0.3, 0.9, 0.3])
VIOLET = np.array([0.6, 0.0, 0.9])
ORANGE = np.array([0.9, 0.6, 0.0])
COLOR_TUPLE = (BLUE, RED, GREEN, VIOLET, ORANGE)


if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any


def render_mesh(
    meshes: Sequence[tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    colorscale: Literal["blue_red", "sequence"] = "sequence",
    interactive: bool = True,
    title: str = "Mesh",
    save_dir: pathlib.Path = pathlib.Path() / "images",
    scalars: npt.NDArray[Any] | None = None,
    cmap: str | Any = "viridis",
    clim: Sequence[float] | None = None,
    show_scalar_bar: bool = False,
) -> None:
    """Renders a sequence of meshes using PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        print(
            "pyvista not installed. Please install it using pip: pip install pyvista."
        )
        return

    plotter = pv.Plotter(
        lighting=None,
        window_size=[1024, 1024],
        notebook=False,
        off_screen=not interactive,
    )
    all_verts = np.concatenate([m[0] for m in meshes if len(m[0]) > 0], axis=0)
    if len(all_verts) > 0:
        min_v = all_verts.min(axis=0)
        max_v = all_verts.max(axis=0)
        center = (min_v + max_v) / 2
        extents = max_v - min_v
        max_extent = extents.max()
        camera_pos = center + np.array(
            [0.8 * max_extent, 0.4 * max_extent, 2.0 * max_extent]
        )
        plotter.camera_position = [camera_pos, center, (0, 1, 0)]
        for light_point in (
            center + np.array([1.5 * max_extent, 1.5 * max_extent, 2 * max_extent]),
            center + np.array([-1.5 * max_extent, 1.5 * max_extent, 2 * max_extent]),
        ):
            light = pv.Light(
                position=light_point,
                focal_point=center,
                intensity=1,
                positional=True,
            )
            plotter.add_light(light)

    for i, (vertices, faces) in enumerate(meshes):
        if not len(vertices):
            continue

        if colorscale == "blue_red":
            i_norm = i / max(1, len(meshes) - 1)
            color = (1 - i_norm) * BLUE + i_norm * RED
        elif colorscale == "sequence":
            color = COLOR_TUPLE[i % len(COLOR_TUPLE)]
        else:
            raise ValueError("Colorscale not available")

        num_faces = faces.shape[0]
        pyvista_faces = np.column_stack(
            [np.full(num_faces, 3, dtype=np.int32), faces]
        ).flatten()
        mesh_pv = pv.PolyData(vertices[:, :3], pyvista_faces)
        if i == 0 and scalars is not None:
            mesh_pv.point_data["scalars"] = scalars
            plotter.add_mesh(
                mesh_pv,
                scalars="scalars",
                cmap=cmap,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                smooth_shading=True,
                show_edges=True,
                edge_color="black",
                line_width=1,
            )
        else:
            plotter.add_mesh(
                mesh_pv,
                color=color,
                smooth_shading=True,
                show_edges=True,
                edge_color="black",
                line_width=1,
            )

    pv.Plotter.enable_eye_dome_lighting(plotter)
    pv.Plotter.enable_shadows(plotter)
    if interactive:
        pv.Plotter.set_background(plotter, color="white")
        plotter.show()
    else:
        save_dir.mkdir(exist_ok=True, parents=True)
        file = save_dir / (title + ".png")
        plotter.screenshot(file, window_size=(1024, 1024), transparent_background=True)

    plotter.close()
    return
