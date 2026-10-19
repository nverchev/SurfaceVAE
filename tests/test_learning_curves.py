"""Tests for the learning curves of the plotting trackers."""

import pytest

from drytorch.trackers.matplotlib import MatPlotter
from drytorch.trackers.plotly import PlotlyPlotter

from src.trackers.learning_curves import save_learning_curves


class TestSaveLearningCurves:
    """Test saving and showing the learning curves of the plotting trackers."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the plotting trackers."""
        self.matplotlib_plotter = mocker.create_autospec(MatPlotter, instance=True)
        self.plotly_plotter = mocker.create_autospec(PlotlyPlotter, instance=True)
        self.trackers = [mocker.Mock(), self.matplotlib_plotter, self.plotly_plotter]
        return

    def test_saves_without_showing(self) -> None:
        """Every plotter saves its figures and none is shown."""
        save_learning_curves(self.trackers, 'model')

        self.matplotlib_plotter.save_figures.assert_called_once_with('model', source_names=('Trainer', 'Validation'))
        self.plotly_plotter.save_figures.assert_called_once_with('model', source_names=('Trainer', 'Validation'))
        self.matplotlib_plotter.plot_grid.assert_not_called()
        self.plotly_plotter.plot.assert_not_called()

    def test_show_opens_matplotlib_in_one_window(self) -> None:
        """Matplotlib shows the metrics as a grid and plotly one figure per metric."""
        save_learning_curves(self.trackers, 'model', show=True)

        self.matplotlib_plotter.plot_grid.assert_called_once_with('model', source_names=('Trainer', 'Validation'))
        self.matplotlib_plotter.plot.assert_not_called()
        self.plotly_plotter.plot.assert_called_once_with('model', source_names=('Trainer', 'Validation'))

    def test_custom_source_names(self) -> None:
        """Custom source names are forwarded to plotters."""
        save_learning_curves(self.trackers, 'model', show=True, source_names=('Trainer',))

        self.matplotlib_plotter.save_figures.assert_called_once_with('model', source_names=('Trainer',))
        self.matplotlib_plotter.plot_grid.assert_called_once_with('model', source_names=('Trainer',))
        self.plotly_plotter.plot.assert_called_once_with('model', source_names=('Trainer',))
