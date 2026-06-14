from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from brainrot_guard.plotbrain_adapter import TribePlotBrainRenderer
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import AnalysisService, PlotBrainFrameService, TribeRuntime
from brainrot_guard.tribe_adapter import TribeV2Predictor
from brainrot_guard.vlm import GatedVLMService
from brainrot_guard.vlm.adapters import HTTPVLMDecomposer, VLMProviderConfig
from brainrot_guard.vlm.config import provider_config_from_env


def build_analysis_service_from_env(
    repository: Repository,
    environ: Mapping[str, str],
    *,
    predictor_factory: Callable[[Mapping[str, str]], object] = TribeV2Predictor.from_env,
    renderer_factory: Callable[[Mapping[str, str]], object] | None = None,
    vlm_service_factory: Callable[[Mapping[str, str]], GatedVLMService | None] | None = None,
) -> AnalysisService | None:
    if not _truthy(environ.get("BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS")):
        return None
    artifacts_raw = environ.get("BRAINROT_GUARD_ARTIFACTS_DIR")
    if not artifacts_raw:
        raise RuntimeError("BRAINROT_GUARD_ARTIFACTS_DIR is required when live analysis is enabled")
    artifacts_dir = Path(artifacts_raw).expanduser().resolve()
    step_ms = _positive_int(environ.get("BRAINROT_GUARD_TRIBE_STEP_MS"), default=1000)
    renderer = (
        renderer_factory(environ)
        if renderer_factory is not None
        else TribePlotBrainRenderer()
    )
    return AnalysisService(
        repository=repository,
        runtime=TribeRuntime(
            predictor=predictor_factory(environ),
            npz_dir=artifacts_dir / "npz",
            step_ms=step_ms,
        ),
        frame_service=PlotBrainFrameService(
            renderer=renderer,
            frames_dir=artifacts_dir / "frames",
        ),
        vlm_service=(vlm_service_factory or build_vlm_service_from_env)(environ),
    )


def build_vlm_service_from_env(
    environ: Mapping[str, str],
    *,
    decomposer_factory: Callable[[VLMProviderConfig], object] = HTTPVLMDecomposer,
) -> GatedVLMService | None:
    if not _truthy(environ.get("BRAINROT_GUARD_ENABLE_VLM")):
        return None
    config = provider_config_from_env(environ)
    threshold = _float(environ.get("BRAINROT_GUARD_VLM_ENGAGEMENT_THRESHOLD"), default=0.8)
    return GatedVLMService(
        decomposer=decomposer_factory(config),
        engagement_threshold=threshold,
        provider=config.provider,
    )


def _positive_int(raw: str | None, *, default: int) -> int:
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise RuntimeError("BRAINROT_GUARD_TRIBE_STEP_MS must be positive")
    return value


def _float(raw: str | None, *, default: float) -> float:
    if raw is None:
        return default
    value = float(raw)
    if not 0 <= value <= 1:
        raise RuntimeError("BRAINROT_GUARD_VLM_ENGAGEMENT_THRESHOLD must be between 0 and 1")
    return value


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["build_analysis_service_from_env", "build_vlm_service_from_env"]
