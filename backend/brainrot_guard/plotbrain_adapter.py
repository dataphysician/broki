from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from brainrot_guard.runtime import NpzPrediction


class TribeRuntimeMissing(RuntimeError):
    pass


def load_plotbrain_plotter(*, importer: Callable[..., Any] = __import__) -> object:
    try:
        plotting = importer("tribev2.plotting", globals(), locals(), ("PlotBrain",), 0)
    except ModuleNotFoundError as exc:
        raise TribeRuntimeMissing(
            "tribev2[plotting] is not installed; install TRIBE v2 with plotting support."
        ) from exc
    plotbrain = getattr(plotting, "PlotBrain", None)
    if not callable(plotbrain):
        raise TribeRuntimeMissing("tribev2.plotting does not expose PlotBrain")
    try:
        return plotbrain(mesh="fsaverage5")
    except TypeError as exc:
        raise TribeRuntimeMissing("tribev2.plotting.PlotBrain must accept mesh='fsaverage5'") from exc


class TribePlotBrainRenderer:
    def __init__(
        self,
        *,
        plotter_loader: Callable[[], object] = load_plotbrain_plotter,
        views: str = "left",
        norm_percentile: int = 100,
        dpi: int = 180,
    ) -> None:
        self.plotter_loader = plotter_loader
        self.views = views
        self.norm_percentile = norm_percentile
        self.dpi = dpi
        self._plotter: object | None = None

    def render_png(self, prediction: NpzPrediction, output_path: Path) -> Path:
        plotter = self._get_plotter()
        plot_timesteps = getattr(plotter, "plot_timesteps", None)
        if not callable(plot_timesteps):
            raise TribeRuntimeMissing("tribev2.plotting.PlotBrain must expose plot_timesteps")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        neuro = np.asarray(prediction.vertex_values, dtype=np.float32).reshape(1, -1)
        figure = plot_timesteps(
            neuro,
            timestamps=[prediction.start_ms / 1000.0],
            plot_every_k_timesteps=1,
            views=self.views,
            norm_percentile=self.norm_percentile,
        )
        try:
            figure.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        except TypeError:
            figure.savefig(output_path)
        finally:
            _close_figure(figure)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"PlotBrain did not create a PNG artifact: {output_path}")
        return output_path

    def _get_plotter(self) -> object:
        if self._plotter is None:
            self._plotter = self.plotter_loader()
        return self._plotter


def _close_figure(figure: object) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.close(figure)


__all__ = ["TribePlotBrainRenderer", "TribeRuntimeMissing", "load_plotbrain_plotter"]
