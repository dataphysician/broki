from pathlib import Path

import numpy as np

from brainrot_guard.app import AnalyzeRequest, create_app
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import AnalysisService, NpzPrediction, PlotBrainFrameService, SegmentWindow, TribePrediction, TribeRuntime
from brainrot_guard.vlm import GatedVLMService, VLMDecompositionRequest


class HighEngagementPredictor:
    def predict_window(self, media, window: SegmentWindow) -> TribePrediction:
        return TribePrediction(
            timestep=window.timestep,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            vertex_values=np.ones(20484, dtype=np.float32),
            attention=0.7,
            engagement=0.93,
            arousal=0.8,
            confidence=0.9,
        )


class FakeRenderer:
    def render_png(self, prediction: NpzPrediction, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_PNG_BYTES)
        return output_path


class FailingDecomposer:
    def decompose(self, request: VLMDecompositionRequest):
        raise RuntimeError("provider quota")


def test_analysis_persists_vlm_not_configured_status(tmp_path: Path) -> None:
    repo, media_id = _repo_with_media(tmp_path)
    service = _analysis_service(tmp_path, repo, vlm_service=None)
    analyze = _endpoint(create_app(repository=repo, analysis_service=service), "/api/media/{media_id}/analyze", {"POST"})

    result = analyze(media_id, AnalyzeRequest(duration_ms=1000))
    status = repo.get_vlm_status(media_id)

    assert result["vlm_status"] == "not_configured"
    assert status["status"] == "not_configured"
    assert status["error"] is None


def test_analysis_persists_vlm_error_status_without_recording_decomposition(tmp_path: Path) -> None:
    repo, media_id = _repo_with_media(tmp_path)
    service = _analysis_service(
        tmp_path,
        repo,
        vlm_service=GatedVLMService(decomposer=FailingDecomposer(), provider="gemini"),
    )
    app = create_app(repository=repo, analysis_service=service)
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    result = analyze(media_id, AnalyzeRequest(duration_ms=1000))
    status = repo.get_vlm_status(media_id)
    response = detail(media_id)

    assert result["vlm_status"] == "error"
    assert status["provider"] == "gemini"
    assert status["error"] == "provider quota"
    assert response["vlm_status"]["status"] == "error"
    assert response["vlm"] is None
    assert response["warning"]["decision"] == "vlm_required"


def _analysis_service(tmp_path: Path, repo: Repository, *, vlm_service) -> AnalysisService:
    return AnalysisService(
        repository=repo,
        runtime=TribeRuntime(predictor=HighEngagementPredictor(), npz_dir=tmp_path / "npz"),
        frame_service=PlotBrainFrameService(renderer=FakeRenderer(), frames_dir=tmp_path / "frames"),
        vlm_service=vlm_service,
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
