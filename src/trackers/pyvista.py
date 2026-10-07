"""PyVista tracker drawing the meshes the events carry."""

import functools
import pathlib

from typing import Any, ClassVar, override

import numpy as np
import pyvista as pv
import vtk

from matplotlib import cm
from numpy import typing as npt

from drytorch.core import log_events
from drytorch.trackers.base_classes import Dumper

from src.config.events import MeshEvent

__all__ = ['MeshTracker']

WINDOW = (1024, 1024)


class MeshTracker(Dumper):
    """Draw every mesh an event carries, to a window or to a file.

    Attributes:
        interactive: Whether a window opens instead of an image being written.
    """

    folder_name: ClassVar[str] = 'images'

    interactive: bool

    def __init__(self, *, interactive: bool = True, par_dir: pathlib.Path | None = None) -> None:
        """Initialize.

        Args:
            interactive: whether a window opens instead of an image being
                written.
            par_dir: directory the images are filed under.
        """
        super().__init__(par_dir=par_dir)
        vtk.vtkObject.GlobalWarningDisplayOff()
        self.interactive = interactive
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: MeshEvent) -> None:
        plotter = pv.Plotter(lighting='none', window_size=list(WINDOW), notebook=False, off_screen=not self.interactive)
        drawn = [np.asarray(vertices) for vertices in event.vertices if len(vertices)]
        if drawn:
            self._light(plotter, np.concatenate(drawn, axis=0))

        triangles = np.insert(event.faces, 0, 3, axis=1).ravel()
        for position, vertices in enumerate(drawn):
            mesh = pv.PolyData(vertices[:, :3], triangles)
            if position or event.scalars is None:
                plotter.add_mesh(mesh, color=event.color, **self._shading(event))
            else:
                self._add_scalars(plotter, mesh, event)

        pv.Plotter.enable_ssao(plotter, kernel_size=256)
        pv.Plotter.enable_shadows(plotter)
        plotter.enable_anti_aliasing('msaa')
        if event.use_edl:
            pv.Plotter.enable_eye_dome_lighting(plotter)

        self._show(plotter, event)
        plotter.close()
        return super().notify(event)

    def _show(self, plotter: pv.Plotter, event: MeshEvent) -> None:
        if self.interactive:
            pv.Plotter.set_background(plotter, color='white')
            plotter.show()
            return

        run_dir = self._get_run_dir()
        directory = run_dir / event.folder if event.folder else run_dir
        directory.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(directory / f'{event.title}.png', window_size=WINDOW, transparent_background=True)
        return

    @staticmethod
    def _add_scalars(plotter: pv.Plotter, mesh: pv.PolyData, event: MeshEvent) -> None:
        scalars = event.scalars
        if scalars is None:
            return

        mesh.point_data['scalars'] = scalars
        shading = MeshTracker._shading(event)
        if scalars.ndim == 2 and scalars.shape[1] == 3:
            plotter.add_mesh(mesh, scalars='scalars', rgb=True, **shading)
            return

        plotter.add_mesh(
            mesh,
            scalars='scalars',
            cmap=cm.get_cmap(event.cmap),
            clim=event.clim,
            show_scalar_bar=event.show_scalar_bar,
            **shading,
        )
        return

    @staticmethod
    def _light(plotter: pv.Plotter, vertices: npt.NDArray[Any]) -> None:
        # three lights placed around the meshes, scaled to how far they spread
        lowest = vertices.min(axis=0)
        highest = vertices.max(axis=0)
        center = (lowest + highest) / 2
        spread = (highest - lowest).max()
        plotter.camera_position = [
            center + np.array([0.8 * spread, 0.4 * spread, 2.0 * spread]),
            center,
            (0, 1, 0),
        ]
        for offset, intensity in (
            (np.array([1.5, 1.5, 2.0]), 1.5),
            (np.array([-1.5, 0.0, 1.5]), 0.5),
            (np.array([0.0, 1.5, -2.0]), 0.8),
        ):
            light = pv.Light(
                position=center + offset * spread,
                focal_point=center,
                intensity=intensity,
                positional=True,
            )
            plotter.add_light(light)

        return

    @staticmethod
    def _shading(event: MeshEvent) -> dict[str, Any]:
        return {
            'smooth_shading': True,
            'show_edges': event.show_edges,
            'edge_color': 'black',
            'line_width': 1,
        }
