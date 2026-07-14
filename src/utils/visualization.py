"""Visualization utilities for meshes using PyVista."""

import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

BONE = np.array([0.89, 0.855, 0.788])

Color = str | Sequence[float] | npt.NDArray[Any]


if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any


def render_mesh(
    vertices: npt.NDArray[Any] | Sequence[npt.NDArray[Any]],
    faces: npt.NDArray[Any],
    color: Color = BONE,
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
    """Renders a sequence of meshes sharing the same topology (faces) using PyVista."""
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
    if isinstance(vertices, np.ndarray) and vertices.ndim == 2:
        vertices_seq = [vertices]
    else:
        vertices_seq = vertices

    all_verts = np.concatenate([v for v in vertices_seq if len(v) > 0], axis=0)
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

    pyvista_faces = np.insert(faces, 0, 3, axis=1).ravel()

    for i, v_raw in enumerate(vertices_seq):
        v = np.asarray(v_raw)
        if not len(v):
            continue

        mesh_pv = pv.PolyData(v[:, :3], pyvista_faces)
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
                color=color,
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
