from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from brainrot_guard.app_types import frame_manifest
from brainrot_guard.artifacts import write_segment_npz
from brainrot_guard.models import MediaItem, SegmentSignal
from brainrot_guard.repository import Repository
from brainrot_guard.vlm import GatedVLMService


class SegmentWindow(BaseModel):
    timestep: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def require_forward_window(self) -> "SegmentWindow":
        if self.end_ms <= self.start_ms:
            raise ValueError("window end_ms must be greater than start_ms")
        return self


class TribePrediction(BaseModel):
    timestep: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    vertex_values: np.ndarray
    attention: float = Field(ge=0, le=1)
    engagement: float = Field(ge=0, le=1)
    arousal: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("vertex_values")
    @classmethod
    def require_vertices(cls, value):
        values = np.asarray(value, dtype=np.float32)
        if values.shape != (20484,):
            raise ValueError("TRIBE fsaverage5 vertex_values must have shape (20484,)")
        return values


class TribePredictor(Protocol):
    def predict_window(self, media: MediaItem, window: SegmentWindow) -> TribePrediction:
        ...


class TribeRuntime:
    def __init__(self, *, predictor: TribePredictor, npz_dir: Path, step_ms: int = 1000) -> None:
        self.predictor = predictor
        self.npz_dir = npz_dir
        self.step_ms = step_ms

    def analyze_media(self, media: MediaItem, *, duration_ms: int) -> list[SegmentSignal]:
        segments: list[SegmentSignal] = []
        for window in plan_segment_windows(duration_ms=duration_ms, step_ms=self.step_ms):
            prediction = self.predictor.predict_window(media, window)
            if (
                prediction.timestep != window.timestep
                or prediction.start_ms != window.start_ms
                or prediction.end_ms != window.end_ms
            ):
                raise ValueError("TRIBE prediction metadata must match requested sampled window")
            npz_path = write_segment_npz(
                self.npz_dir,
                media_id=media.id,
                timestep=prediction.timestep,
                start_ms=prediction.start_ms,
                end_ms=prediction.end_ms,
                vertex_values=prediction.vertex_values,
            )
            segments.append(
                SegmentSignal(
                    timestep=prediction.timestep,
                    start_ms=prediction.start_ms,
                    end_ms=prediction.end_ms,
                    attention=prediction.attention,
                    engagement=prediction.engagement,
                    arousal=prediction.arousal,
                    confidence=prediction.confidence,
                    npz_path=npz_path,
                )
            )
        return segments


def plan_segment_windows(*, duration_ms: int, step_ms: int = 1000) -> list[SegmentWindow]:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    windows: list[SegmentWindow] = []
    timestep = 0
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(start_ms + step_ms, duration_ms)
        windows.append(SegmentWindow(timestep=timestep, start_ms=start_ms, end_ms=end_ms))
        start_ms = end_ms
        timestep += 1
    return windows


class NpzPrediction(BaseModel):
    timestep: int
    start_ms: int
    end_ms: int
    mesh: str
    vertex_values: np.ndarray

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("mesh")
    @classmethod
    def require_mesh(cls, value: str) -> str:
        if value != "fsaverage5":
            raise ValueError("PlotBrain prediction artifacts must use mesh='fsaverage5'")
        return value

    @field_validator("vertex_values")
    @classmethod
    def require_vertices(cls, value):
        values = np.asarray(value, dtype=np.float32)
        if values.shape != (20484,):
            raise ValueError("PlotBrain prediction artifacts must contain 20484 vertices")
        return values


class PlotBrainRenderer(Protocol):
    def render_png(self, prediction: NpzPrediction, output_path: Path) -> Path:
        ...


class PlotBrainFrameService:
    def __init__(self, *, renderer: PlotBrainRenderer, frames_dir: Path) -> None:
        self.renderer = renderer
        self.frames_dir = frames_dir

    def render_missing_frames(self, media_id: str, segments: list[SegmentSignal]) -> list[SegmentSignal]:
        rendered: list[SegmentSignal] = []
        for segment in segments:
            if segment.frame_path is not None and segment.frame_path.exists():
                rendered.append(segment)
                continue
            prediction = load_npz_prediction(segment.npz_path)
            if prediction.timestep != segment.timestep or prediction.start_ms != segment.start_ms or prediction.end_ms != segment.end_ms:
                raise ValueError("PlotBrain prediction artifact metadata must match segment row")
            output = self.frames_dir / media_id / f"{segment.timestep:06d}.png"
            rendered.append(segment.model_copy(update={"frame_path": self.renderer.render_png(prediction, output)}))
        return rendered


def load_npz_prediction(path: Path) -> NpzPrediction:
    if not path.exists():
        raise FileNotFoundError(f"missing TRIBE prediction artifact: {path}")
    with np.load(path) as data:
        return NpzPrediction(
            timestep=int(data["timestep"]),
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            mesh=str(data["mesh"]),
            vertex_values=data["vertex_values"],
        )


@dataclass(frozen=True)
class AnalysisService:
    repository: Repository
    runtime: TribeRuntime
    frame_service: PlotBrainFrameService
    vlm_service: GatedVLMService | None = None

    def analyze(self, media_id: str, *, duration_ms: int) -> dict:
        media = self.repository.get_media(media_id)
        if media is None:
            raise ValueError(f"unknown media id: {media_id}")
        segments = self.runtime.analyze_media(media, duration_ms=duration_ms)
        rendered = self.frame_service.render_missing_frames(media_id, segments)
        self.repository.upsert_segments(media_id, rendered)
        vlm_status = "not_configured"
        if self.vlm_service is not None:
            vlm_result = self.vlm_service.maybe_decompose(media, rendered)
            vlm_status = vlm_result.status
            if vlm_result.decomposition is not None:
                self.repository.record_vlm(media_id, vlm_result.decomposition)
            self.repository.record_vlm_status(
                media_id,
                status=vlm_result.status,
                provider=vlm_result.provider,
                error=vlm_result.error,
            )
        else:
            self.repository.record_vlm_status(media_id, status=vlm_status)
        return {
            "media_id": media_id,
            "segment_count": len(rendered),
            "frame_manifest": frame_manifest(media_id, rendered),
            "vlm_status": vlm_status,
        }
