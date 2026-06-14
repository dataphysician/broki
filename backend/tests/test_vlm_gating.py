from pathlib import Path

import numpy as np

from brainrot_guard.app import AnalyzeRequest, create_app
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import VLMDecomposition
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import AnalysisService, NpzPrediction, PlotBrainFrameService, SegmentWindow, TribePrediction, TribeRuntime
from brainrot_guard.vlm import VLMDecompositionRequest, GatedVLMService


class EngagementPredictor:
    def __init__(self, engagement: float) -> None:
        self.engagement = engagement

    def predict_window(self, media, window: SegmentWindow) -> TribePrediction:
        return TribePrediction(
            timestep=window.timestep,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            vertex_values=np.ones(20484, dtype=np.float32),
            attention=0.6,
            engagement=self.engagement,
            arousal=0.7,
            confidence=0.9,
        )


class FakeRenderer:
    def render_png(self, prediction: NpzPrediction, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_PNG_BYTES)
        return output_path


class CountingDecomposer:
    def __init__(self) -> None:
        self.calls: list[VLMDecompositionRequest] = []

    def decompose(self, request: VLMDecompositionRequest) -> VLMDecomposition:
        self.calls.append(request)
        return VLMDecomposition(
            theme="toy unboxing",
            pacing_score=0.93,
            scene_change_cadence_hz=0.8,
            contrast_score=0.85,
            sound_effect_density=0.75,
            educational_value=0.08,
            emotional_hook_score=0.88,
            novelty_score=0.8,
            repetition_score=0.9,
            risk_score=0.91,
            risk_rationale="high sensory reward density and low learning value",
        )


def test_low_engagement_analysis_does_not_call_vlm(tmp_path: Path) -> None:
    repo, media_id = _repo_with_media(tmp_path)
    decomposer = CountingDecomposer()
    service = _analysis_service(tmp_path, repo, engagement=0.3, decomposer=decomposer)
    app = create_app(repository=repo, analysis_service=service)
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    analyze(media_id, AnalyzeRequest(duration_ms=1000))
    response = detail(media_id)

    assert decomposer.calls == []
    assert response["vlm"] is None
    assert response["warning"]["decision"] == "allow"


def test_high_engagement_analysis_calls_vlm_and_persists_structured_decomposition(tmp_path: Path) -> None:
    repo, media_id = _repo_with_media(tmp_path)
    decomposer = CountingDecomposer()
    service = _analysis_service(tmp_path, repo, engagement=0.92, decomposer=decomposer)
    app = create_app(repository=repo, analysis_service=service)
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    result = analyze(media_id, AnalyzeRequest(duration_ms=1000))
    response = detail(media_id)

    assert result["vlm_status"] == "complete"
    assert len(decomposer.calls) == 1
    assert decomposer.calls[0].media.id == media_id
    assert decomposer.calls[0].frame_paths[0].name == "000000.png"
    assert response["vlm"]["theme"] == "toy unboxing"
    assert response["vlm"]["risk_score"] == 0.91
    assert response["warning"]["decision"] == "warning_ready"


def test_high_engagement_without_decomposer_remains_vlm_required(tmp_path: Path) -> None:
    repo, media_id = _repo_with_media(tmp_path)
    service = _analysis_service(tmp_path, repo, engagement=0.92, decomposer=None)
    app = create_app(repository=repo, analysis_service=service)
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    result = analyze(media_id, AnalyzeRequest(duration_ms=1000))
    response = detail(media_id)

    assert result["vlm_status"] == "not_configured"
    assert response["vlm"] is None
    assert response["warning"]["decision"] == "vlm_required"


def _analysis_service(tmp_path: Path, repo: Repository, *, engagement: float, decomposer) -> AnalysisService:
    return AnalysisService(
        repository=repo,
        runtime=TribeRuntime(predictor=EngagementPredictor(engagement), npz_dir=tmp_path / "npz"),
        frame_service=PlotBrainFrameService(renderer=FakeRenderer(), frames_dir=tmp_path / "frames"),
        vlm_service=GatedVLMService(decomposer=decomposer) if decomposer is not None else None,
    )


def _repo_with_media(tmp_path: Path) -> tuple[Repository, str]:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "story.txt").write_text("local text", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    repo.upsert_media(media)
    return repo, media.id


def _endpoint(app, path: str, methods: set[str]):
    for route in app.routes:
        if getattr(route, "path", None) == path and getattr(route, "methods", set()) == methods:
            return route.endpoint
    raise AssertionError(f"route not found: {methods} {path}")


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)
