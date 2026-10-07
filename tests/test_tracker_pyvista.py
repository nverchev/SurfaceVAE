"""Tests for the mesh tracker."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.config import Experiment, MeshEvent
from src.config.hydra import get_config_all
from src.trackers.pyvista import MeshTracker

FACES = np.array([[0, 1, 2]])
VERTICES = np.zeros((3, 3))


def test_mesh_event_notifies_the_tracker() -> None:
    """Publishing a mesh event draws it through a plotter."""
    exp = Experiment(get_config_all(), par_dir='/tmp/experiments')
    tracker = MeshTracker(interactive=False)
    exp.trackers.subscribe(tracker)

    with exp.create_run(record=False), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh')

        plotter_cls.return_value.screenshot.assert_called_once()
        plotter_cls.return_value.close.assert_called_once()


def test_the_folder_of_an_event_is_a_subdirectory() -> None:
    """A mesh event files its drawing under the folder it names."""
    exp = Experiment(get_config_all(), par_dir='/tmp/experiments')
    tracker = MeshTracker(interactive=False)
    exp.trackers.subscribe(tracker)

    with exp.create_run(record=False), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh', folder='sample_0')
        written = plotter_cls.return_value.screenshot.call_args.args[0]

        assert written.parent.name == 'sample_0'
        assert written.name == 'a_mesh.png'
        assert written.parent.is_dir()


def test_an_interactive_tracker_opens_a_window() -> None:
    """An interactive tracker shows the meshes instead of writing them."""
    exp = Experiment(get_config_all(), par_dir='/tmp/experiments')
    tracker = MeshTracker(interactive=True)
    exp.trackers.subscribe(tracker)

    with exp.create_run(record=False), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh')

        plotter_cls.return_value.show.assert_called_once()
        plotter_cls.return_value.screenshot.assert_not_called()
