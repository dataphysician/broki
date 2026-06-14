from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class MediaKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"


class MediaItem(BaseModel):
    id: str
    path: Path
    kind: MediaKind
    mime_type: str
    duration_ms: int | None = Field(default=None, gt=0)
    source: str = "local_folder"


class SegmentSignal(BaseModel):
    timestep: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    attention: float = Field(ge=0, le=1)
    engagement: float = Field(ge=0, le=1)
    arousal: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    mesh: str = "fsaverage5"
    vertex_count: int = 20484
    npz_path: Path
    frame_path: Path | None = None

    @field_validator("mesh")
    @classmethod
    def require_mesh(cls, value: str) -> str:
        if value != "fsaverage5":
            raise ValueError("segment predictions must use mesh='fsaverage5'")
        return value

    @field_validator("vertex_count")
    @classmethod
    def require_vertex_count(cls, value: int) -> int:
        if value != 20484:
            raise ValueError("fsaverage5 prediction vectors must contain 20484 vertices")
        return value

    @model_validator(mode="after")
    def require_forward_window(self) -> "SegmentSignal":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class VLMDecomposition(BaseModel):
    theme: str
    pacing_score: float = Field(ge=0, le=1)
    scene_change_cadence_hz: float = Field(ge=0)
    contrast_score: float = Field(ge=0, le=1)
    sound_effect_density: float = Field(ge=0, le=1)
    educational_value: float = Field(ge=0, le=1)
    emotional_hook_score: float = Field(ge=0, le=1)
    novelty_score: float = Field(ge=0, le=1)
    repetition_score: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    risk_rationale: str
