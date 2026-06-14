from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from brainrot_guard.models import MediaItem, MediaKind
from brainrot_guard.runtime import SegmentWindow, TribePrediction


DEFAULT_TRIBE_MODEL_REF = "Jessylg27/tribev2-lite-qv"


class TribeAdapterError(RuntimeError):
    pass


def load_tribe_model(
    model_ref: str = DEFAULT_TRIBE_MODEL_REF,
    cache_folder: Path | None = None,
    *,
    importer: Callable[..., Any] = __import__,
) -> object:
    model_cls = _import_tribe_model(importer)
    try:
        return model_cls.from_pretrained(model_ref, cache_folder=cache_folder)
    except TypeError:
        return model_cls.from_pretrained(model_ref)


class TribeV2Predictor:
    def __init__(
        self,
        *,
        model_loader: Callable[[], object],
        image_video_dir: Path | None = None,
        image_video_factory: Callable[[MediaItem, Path], Path] | None = None,
    ) -> None:
        self.model_loader = model_loader
        self.image_video_dir = image_video_dir
        self.image_video_factory = image_video_factory or self._generate_image_video
        self._model: object | None = None
        self._prediction_cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "TribeV2Predictor":
        model_ref = (
            environ.get("BRAINROT_GUARD_TRIBE_MODEL_ID")
            or environ.get("BRAINROT_GUARD_TRIBE_CKPT")
            or DEFAULT_TRIBE_MODEL_REF
        )
        cache_raw = environ.get("BRAINROT_GUARD_TRIBE_CACHE_DIR")
        cache_folder = Path(cache_raw) if cache_raw else None
        image_video_raw = environ.get("BRAINROT_GUARD_IMAGE_VIDEO_DIR")
        artifacts_raw = environ.get("BRAINROT_GUARD_ARTIFACTS_DIR")
        image_video_dir = (
            Path(image_video_raw)
            if image_video_raw
            else (Path(artifacts_raw) / "image-videos" if artifacts_raw else None)
        )
        return cls(
            model_loader=lambda: load_tribe_model(model_ref, cache_folder),
            image_video_dir=image_video_dir,
        )

    def predict_window(self, media: MediaItem, window: SegmentWindow) -> TribePrediction:
        predictions = self._predict_media(media)
        if window.timestep >= predictions.shape[0]:
            raise TribeAdapterError(
                f"TRIBE did not return timestep {window.timestep} for {media.path.name}"
            )
        values = np.asarray(predictions[window.timestep], dtype=np.float32)
        attention, engagement, arousal, confidence = _proxy_scores(values)
        return TribePrediction(
            timestep=window.timestep,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            vertex_values=values,
            attention=attention,
            engagement=engagement,
            arousal=arousal,
            confidence=confidence,
        )

    def _predict_media(self, media: MediaItem) -> np.ndarray:
        cached = self._prediction_cache.get(media.id)
        if cached is not None:
            return cached
        model = self._get_model()
        events = model.get_events_dataframe(**self._events_kwargs(media))
        preds, _segments = model.predict(events=events)
        predictions = _as_prediction_array(preds)
        self._prediction_cache[media.id] = predictions
        return predictions

    def _get_model(self) -> object:
        if self._model is None:
            self._model = self.model_loader()
        return self._model

    def _events_kwargs(self, media: MediaItem) -> dict[str, Path]:
        if media.kind == MediaKind.VIDEO:
            return {"video_path": media.path}
        if media.kind == MediaKind.AUDIO:
            return {"audio_path": media.path}
        if media.kind == MediaKind.TEXT:
            return {"text_path": media.path}
        if media.kind == MediaKind.IMAGE:
            return {"video_path": self._image_video_path(media)}
        raise TribeAdapterError(f"unsupported media kind for TRIBE inference: {media.kind}")

    def _image_video_path(self, media: MediaItem) -> Path:
        if self.image_video_dir is None:
            raise TribeAdapterError("image_video_dir is required for live TRIBE image analysis")
        output_path = self.image_video_dir / f"{media.id}.mp4"
        if output_path.exists():
            return output_path
        return self.image_video_factory(media, output_path)

    @staticmethod
    def _generate_image_video(media: MediaItem, output_path: Path) -> Path:
        duration_ms = media.duration_ms or 3000
        return generate_static_image_video(media.path, output_path, duration_ms=duration_ms)


def _import_tribe_model(importer: Callable[..., Any]) -> object:
    try:
        module = importer("tribev2", globals(), locals(), ("TribeModel",), 0)
        return module.TribeModel
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass
    try:
        module = importer("tribev2.demo_utils", globals(), locals(), ("TribeModel",), 0)
        return module.TribeModel
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise TribeAdapterError(
            "tribev2 package is not installed or does not expose TribeModel"
        ) from exc


def _as_prediction_array(preds: object) -> np.ndarray:
    if hasattr(preds, "detach"):
        preds = preds.detach()
    if hasattr(preds, "cpu"):
        preds = preds.cpu()
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    values = np.asarray(preds, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 20484:
        raise TribeAdapterError("TRIBE predictions must have shape (n_timesteps, 20484)")
    return values


def _proxy_scores(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(values)
    confidence = float(finite.mean())
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean_abs = float(np.mean(np.abs(clean)))
    high_abs = float(np.percentile(np.abs(clean), 95))
    spread = float(np.std(clean))
    attention = _squash(mean_abs)
    engagement = _squash(high_abs)
    arousal = _squash(spread)
    return attention, engagement, arousal, confidence


def _squash(value: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - np.exp(-max(0.0, value)))))


def generate_static_image_video(
    image_path: Path,
    output_path: Path,
    *,
    duration_ms: int = 3000,
    runner: Callable[..., object] = subprocess.run,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-vf",
        "scale=max(2\\,trunc(iw/2)*2):max(2\\,trunc(ih/2)*2),format=yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = runner(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise TribeAdapterError("ffmpeg is required for live TRIBE image analysis") from exc
    if getattr(result, "returncode", 1) != 0:
        stderr = getattr(result, "stderr", "")
        raise TribeAdapterError(f"ffmpeg failed to create static image video: {stderr}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TribeAdapterError(f"ffmpeg did not create image video artifact: {output_path}")
    return output_path


__all__ = [
    "DEFAULT_TRIBE_MODEL_REF",
    "TribeAdapterError",
    "TribeV2Predictor",
    "generate_static_image_video",
    "load_tribe_model",
]
