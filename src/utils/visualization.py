"""Visualization utilities for meshes using PyVista."""

import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

TERRACOTTA = np.array([0.75, 0.45, 0.35])


if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any


def render_mesh(
    meshes: Sequence[tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    color: str | Sequence[float] | Sequence[str | Sequence[float]] | None = None,
    interactive: bool = True,
    title: str = "Mesh",
    save_dir: pathlib.Path = pathlib.Path() / "images",
    scalars: npt.NDArray[Any] | None = None,
    cmap: str | Any = "viridis",
    clim: Sequence[float] | None = None,
    show_scalar_bar: bool = False,
    show_edges: bool = False,
    use_edl: bool = False,
) -> None:
    """Renders a sequence of meshes using PyVista."""
    try:
        import pyvista as pv
        import vtk

        vtk.vtkObject.GlobalWarningDisplayOff()

    except ImportError:
        print(
            "pyvista not installed. Please install it using pip: pip install pyvista."
        )
        return

    import matplotlib.cm as cm

    plotter = pv.Plotter(
        lighting="none",
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
        key_light = pv.Light(
            position=center
            + np.array([1.5 * max_extent, 1.5 * max_extent, 2.0 * max_extent]),
            focal_point=center,
            intensity=1.5,
            positional=True,
        )
        plotter.add_light(key_light)
        fill_light = pv.Light(
            position=center + np.array([-1.5 * max_extent, 0.0, 1.5 * max_extent]),
            focal_point=center,
            intensity=0.5,
            positional=True,
        )
        plotter.add_light(fill_light)
        back_light = pv.Light(
            position=center + np.array([0.0, 1.5 * max_extent, -2.0 * max_extent]),
            focal_point=center,
            intensity=0.8,
            positional=True,
        )
        plotter.add_light(back_light)

    for i, (vertices, faces) in enumerate(meshes):
        if not len(vertices):
            continue

        if color is None:
            mesh_color: Any = TERRACOTTA
        elif isinstance(color, str):
            mesh_color = color
        elif isinstance(color, Sequence) and not isinstance(
            color[0], (int, float, np.integer, np.floating)
        ):
            mesh_color = color[i % len(color)]
        else:
            mesh_color = color

        num_faces = faces.shape[0]
        pyvista_faces = np.column_stack(
            [np.full(num_faces, 3, dtype=np.int32), faces]
        ).flatten()
        mesh_pv = pv.PolyData(vertices[:, :3], pyvista_faces)
        if i == 0 and scalars is not None:
            mesh_pv.point_data["scalars"] = scalars
            is_rgb = scalars.ndim == 2 and scalars.shape[1] == 3
            if is_rgb:
                plotter.add_mesh(
                    mesh_pv,
                    scalars="scalars",
                    rgb=True,
                    smooth_shading=True,
                    show_edges=show_edges,
                    edge_color="black",
                    line_width=1,
                )
            else:
                plotter.add_mesh(
                    mesh_pv,
                    scalars="scalars",
                    cmap=cm.get_cmap(cmap),
                    clim=clim,
                    show_scalar_bar=show_scalar_bar,
                    smooth_shading=True,
                    show_edges=show_edges,
                    edge_color="black",
                    line_width=1,
                )
        else:
            plotter.add_mesh(
                mesh_pv,
                color=mesh_color,
                smooth_shading=True,
                show_edges=show_edges,
                edge_color="black",
                line_width=1,
            )

    pv.Plotter.enable_ssao(plotter, kernel_size=256)
    pv.Plotter.enable_shadows(plotter)
    plotter.enable_anti_aliasing("msaa")
    if use_edl:
        pv.Plotter.enable_eye_dome_lighting(plotter)

    if interactive:
        pv.Plotter.set_background(plotter, color="white")
        plotter.show()
    else:
        save_dir.mkdir(exist_ok=True, parents=True)
        file = save_dir / (title + ".png")
        plotter.screenshot(file, window_size=(1024, 1024), transparent_background=True)

    plotter.close()
    return
