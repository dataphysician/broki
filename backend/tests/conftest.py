from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brainrot_guard.artifacts import write_segment_npz
from brainrot_guard.models import VLMDecomposition
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import (
    NpzPrediction,
    SegmentWindow,
    TribePrediction,
    TribeRuntime,
)
from brainrot_guard.vlm import GatedVLMService, VLMDecompositionRequest


_TMP_DB_PATH_ENV = "BROKI_TEST_DB_PATH"
_SAMPLE_FIXTURE_TEXT = "test fixture"
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)
_VERTICES = np.ones(20484, dtype=np.float32)


class FakePredictor:
    def __init__(self, engagement: float = 0.91, attention: float = 0.6, arousal: float = 0.7, confidence: float = 0.9) -> None:
        self.engagement = engagement
        self.attention = attention
        self.arousal = arousal
        self.confidence = confidence

    def predict_window(self, media, window: SegmentWindow) -> TribePrediction:
        return TribePrediction(
            timestep=window.timestep,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            vertex_values=_VERTICES,
            attention=self.attention,
            engagement=self.engagement,
            arousal=self.arousal,
            confidence=self.confidence,
        )


class FakeRenderer:
    def render_png(self, prediction: NpzPrediction, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_PNG_BYTES)
        return output_path


class FakeDecomposer:
    def __init__(self, *, risk_score: float = 0.7, risk_rationale: str = "test fixture") -> None:
        self.risk_score = risk_score
        self.risk_rationale = risk_rationale
        self.calls: list[VLMDecompositionRequest] = []

    def decompose(self, request: VLMDecompositionRequest) -> VLMDecomposition:
        self.calls.append(request)
        return VLMDecomposition(
            theme="test fixture",
            pacing_score=0.9,
            scene_change_cadence_hz=0.8,
            contrast_score=0.85,
            sound_effect_density=0.75,
            educational_value=0.08,
            emotional_hook_score=0.88,
            novelty_score=0.8,
            repetition_score=0.9,
            risk_score=self.risk_score,
            risk_rationale=self.risk_rationale,
        )


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Repository:
    db_path = tmp_path / "state.sqlite3"
    repo = Repository(db_path)
    repo.initialize()
    return repo


@pytest.fixture()
def sample_media_dir(tmp_path: Path) -> Path:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "story.txt").write_text(_SAMPLE_FIXTURE_TEXT, encoding="utf-8")
    (media_dir / "frame.png").write_bytes(_PNG_BYTES)
    (media_dir / "audio.wav").write_bytes(b"RIFF\x00")
    (media_dir / "track.mp3").write_bytes(b"ID3")
    (media_dir / "link.url").write_text("https://example.com", encoding="utf-8")
    return media_dir


@pytest.fixture()
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture()
def fake_renderer() -> FakeRenderer:
    return FakeRenderer()


@pytest.fixture()
def fake_decomposer() -> FakeDecomposer:
    return FakeDecomposer()


@pytest.fixture()
def gated_vlm_service(fake_decomposer: FakeDecomposer) -> GatedVLMService:
    return GatedVLMService(decomposer=fake_decomposer)


@pytest.fixture()
def tribe_runtime(tmp_path: Path, fake_predictor: FakePredictor) -> TribeRuntime:
    return TribeRuntime(predictor=fake_predictor, npz_dir=tmp_path / "npz")


@pytest.fixture()
def plotbrain_frame_service(tmp_path: Path, fake_renderer: FakeRenderer):
    from brainrot_guard.runtime import PlotBrainFrameService

    return PlotBrainFrameService(renderer=fake_renderer, frames_dir=tmp_path / "frames")


@pytest.fixture()
def segment_npz(tmp_path: Path) -> Path:
    return write_segment_npz(
        tmp_path,
        media_id="media1",
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=_VERTICES,
    )
