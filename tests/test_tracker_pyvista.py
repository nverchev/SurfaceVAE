"""Tests for the mesh tracker."""

import pathlib

from unittest.mock import MagicMock, patch

import numpy as np

from src.config import Experiment
from src.lib.events import MeshEvent
from src.config.hydra import get_experiment_config
from src.trackers.pyvista import MeshTracker

FACES = np.array([[0, 1, 2]])
VERTICES = np.zeros((3, 3))


def test_mesh_event_notifies_the_tracker(tmp_path: pathlib.Path) -> None:
    """Publishing a mesh event draws it through a plotter."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    tracker = MeshTracker()
    exp.trackers.subscribe(tracker)

    with exp.create_run(), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh')

        plotter_cls.return_value.screenshot.assert_called_once()
        plotter_cls.return_value.close.assert_called_once()


def test_the_folder_of_an_event_is_a_subdirectory(tmp_path: pathlib.Path) -> None:
    """A mesh event files its drawing under the folder it names."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    tracker = MeshTracker()
    exp.trackers.subscribe(tracker)

    with exp.create_run(), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh', folder='sample_0')
        written = plotter_cls.return_value.screenshot.call_args.args[0]

        assert written.parent.name == 'sample_0'
        assert written.name == 'a_mesh.png'
        assert written.parent.is_dir()


def test_a_shown_mesh_is_also_written(tmp_path: pathlib.Path) -> None:
    """A tracker that shows the meshes still writes them for a recorded run."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    tracker = MeshTracker(show=True)
    exp.trackers.subscribe(tracker)

    with exp.create_run(), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh')

        plotter_cls.return_value.show.assert_called_once()
        plotter_cls.return_value.screenshot.assert_called_once()


def test_an_unrecorded_run_is_only_shown(tmp_path: pathlib.Path) -> None:
    """A tracker that shows the meshes writes nothing for an unrecorded run."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    tracker = MeshTracker(show=True)
    exp.trackers.subscribe(tracker)

    with exp.create_run(record=False), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh')

        plotter_cls.return_value.show.assert_called_once()
        plotter_cls.return_value.screenshot.assert_not_called()


def test_an_unrecorded_run_writes_no_screenshot(tmp_path: pathlib.Path) -> None:
    """A tracker that does not show the meshes draws nothing for an unrecorded run."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    tracker = MeshTracker()
    exp.trackers.subscribe(tracker)

    with exp.create_run(record=False), patch('pyvista.Plotter') as plotter_cls:
        plotter_cls.return_value = MagicMock()
        MeshEvent(vertices=[VERTICES], faces=FACES, title='a_mesh')

        plotter_cls.return_value.screenshot.assert_not_called()
