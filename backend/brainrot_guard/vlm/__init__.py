from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from brainrot_guard.models import MediaItem, SegmentSignal, VLMDecomposition


@dataclass(frozen=True)
class VLMDecompositionRequest:
    media: MediaItem
    segments: list[SegmentSignal]
    frame_paths: tuple[Path, ...]


class VLMDecomposer(Protocol):
    def decompose(self, request: VLMDecompositionRequest) -> VLMDecomposition:
        ...


@dataclass(frozen=True)
class GatedVLMResult:
    status: str
    decomposition: VLMDecomposition | None
    provider: str | None = None
    error: str | None = None


class GatedVLMService:
    def __init__(
        self,
        *,
        decomposer: VLMDecomposer,
        engagement_threshold: float = 0.8,
        provider: str = "configured",
    ) -> None:
        self.decomposer = decomposer
        self.engagement_threshold = engagement_threshold
        self.provider = provider

    def maybe_decompose(self, media: MediaItem, segments: list[SegmentSignal]) -> GatedVLMResult:
        if not any(segment.engagement >= self.engagement_threshold for segment in segments):
            return GatedVLMResult(status="skipped_engagement_gate", decomposition=None, provider=self.provider)
        frame_paths = tuple(segment.frame_path for segment in segments if segment.frame_path is not None)
        try:
            decomposition = self.decomposer.decompose(
                VLMDecompositionRequest(media=media, segments=segments, frame_paths=frame_paths)
            )
        except Exception as exc:
            return GatedVLMResult(
                status="error",
                decomposition=None,
                provider=self.provider,
                error=str(exc),
            )
        return GatedVLMResult(status="complete", decomposition=decomposition, provider=self.provider)
