from pathlib import Path

import numpy as np
import pytest

from brainrot_guard.runtime import NpzPrediction


def test_tribe_plotbrain_renderer_saves_single_timestep_png(tmp_path: Path) -> None:
    from brainrot_guard.plotbrain_adapter import TribePlotBrainRenderer

    plotter = FakePlotBrain()
    renderer = TribePlotBrainRenderer(plotter_loader=lambda: plotter)
    prediction = _prediction()

    output = renderer.render_png(prediction, tmp_path / "frames" / "000002.png")

    assert output == tmp_path / "frames" / "000002.png"
    assert output.read_bytes().startswith(b"\x89PNG")
    assert plotter.neuro.shape == (1, 20484)
    assert plotter.timestamps == [2.0]
    assert plotter.mesh == "fsaverage5"


def test_tribe_plotbrain_renderer_requires_plot_timesteps(tmp_path: Path) -> None:
    from brainrot_guard.plotbrain_adapter import TribePlotBrainRenderer, TribeRuntimeMissing

    renderer = TribePlotBrainRenderer(plotter_loader=lambda: object())

    with pytest.raises(TribeRuntimeMissing, match="plot_timesteps"):
        renderer.render_png(_prediction(), tmp_path / "frame.png")


def test_load_plotbrain_plotter_instantiates_fsaverage5_plotbrain() -> None:
    from brainrot_guard.plotbrain_adapter import load_plotbrain_plotter

    calls = []

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "tribev2.plotting"
        assert fromlist == ("PlotBrain",)

        class Module:
            @staticmethod
            def PlotBrain(*, mesh: str):
                calls.append(mesh)
                return FakePlotBrain(mesh=mesh)

        return Module()

    plotter = load_plotbrain_plotter(importer=importer)

    assert isinstance(plotter, FakePlotBrain)
    assert calls == ["fsaverage5"]


def test_load_plotbrain_plotter_fails_clearly_when_plotting_extra_is_missing() -> None:
    from brainrot_guard.plotbrain_adapter import TribeRuntimeMissing, load_plotbrain_plotter

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        raise ModuleNotFoundError("No module named 'tribev2.plotting'")

    with pytest.raises(TribeRuntimeMissing, match=r"tribev2\[plotting\]"):
        load_plotbrain_plotter(importer=importer)


class FakePlotBrain:
    def __init__(self, *, mesh: str = "fsaverage5") -> None:
        self.mesh = mesh
        self.neuro = None
        self.timestamps = None

    def plot_timesteps(self, neuro, *, timestamps, **kwargs):
        self.neuro = neuro
        self.timestamps = timestamps
        return FakeFigure()


class FakeFigure:
    def savefig(self, output_path: Path, **kwargs) -> None:
        output_path.write_bytes(_PNG_BYTES)


def _prediction() -> NpzPrediction:
    return NpzPrediction(
        timestep=2,
        start_ms=2000,
        end_ms=3000,
        mesh="fsaverage5",
        vertex_values=np.ones(20484, dtype=np.float32),
    )


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)
