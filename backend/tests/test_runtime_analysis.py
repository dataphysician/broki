from pathlib import Path

import numpy as np

from brainrot_guard.app import AnalyzeRequest, create_app
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import (
    AnalysisService,
    NpzPrediction,
    PlotBrainFrameService,
    SegmentWindow,
    TribePrediction,
    TribeRuntime,
    plan_segment_windows,
)


class FakePredictor:
    def __init__(self) -> None:
        self.windows: list[SegmentWindow] = []

    def predict_window(self, media, window: SegmentWindow) -> TribePrediction:
        self.windows.append(window)
        return TribePrediction(
            timestep=window.timestep,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            vertex_values=np.ones(20484, dtype=np.float32) * (window.timestep + 1) / 10,
            attention=0.6,
            engagement=0.91,
            arousal=0.7,
            confidence=0.9,
        )


class FakeRenderer:
    def __init__(self) -> None:
        self.predictions: list[NpzPrediction] = []

    def render_png(self, prediction: NpzPrediction, output_path: Path) -> Path:
        self.predictions.append(prediction)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_PNG_BYTES)
        return output_path


def test_plan_segment_windows_samples_media_duration() -> None:
    windows = plan_segment_windows(duration_ms=2500, step_ms=1000)

    assert [(w.timestep, w.start_ms, w.end_ms) for w in windows] == [
        (0, 0, 1000),
        (1, 1000, 2000),
        (2, 2000, 2500),
    ]


def test_tribe_runtime_persists_window_npz_artifacts(tmp_path: Path) -> None:
    media = _media(tmp_path)
    predictor = FakePredictor()
    runtime = TribeRuntime(predictor=predictor, npz_dir=tmp_path / "npz")

    segments = runtime.analyze_media(media, duration_ms=2100)

    assert len(segments) == 3
    assert len(predictor.windows) == 3
    assert all(segment.npz_path.exists() for segment in segments)
    with np.load(segments[1].npz_path) as data:
        assert int(data["timestep"]) == 1
        assert int(data["start_ms"]) == 1000
        assert int(data["end_ms"]) == 2000
        assert data["vertex_values"].shape == (20484,)


def test_plotbrain_frame_service_renders_only_from_npz_prediction_artifacts(tmp_path: Path) -> None:
    media = _media(tmp_path)
    segments = TribeRuntime(predictor=FakePredictor(), npz_dir=tmp_path / "npz").analyze_media(media, duration_ms=1000)
    renderer = FakeRenderer()
    service = PlotBrainFrameService(renderer=renderer, frames_dir=tmp_path / "frames")

    rendered = service.render_missing_frames(media.id, segments)

    assert rendered[0].frame_path is not None
    assert rendered[0].frame_path.exists()
    assert renderer.predictions[0].mesh == "fsaverage5"
    assert renderer.predictions[0].vertex_values.shape == (20484,)


def test_analysis_api_fails_clearly_until_runtime_is_configured(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media = _media(tmp_path)
    repo.upsert_media(media)
    app = create_app(repository=repo)
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})

    try:
        analyze(media.id, AnalyzeRequest(duration_ms=1000))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 503
        assert "analysis runtime is not configured" in str(exc)
    else:
        raise AssertionError("unconfigured runtime must fail clearly")


def test_analysis_api_runs_runtime_and_records_rendered_frames(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media = _media(tmp_path)
    repo.upsert_media(media)
    runtime = TribeRuntime(predictor=FakePredictor(), npz_dir=tmp_path / "npz")
    frame_service = PlotBrainFrameService(renderer=FakeRenderer(), frames_dir=tmp_path / "frames")
    app = create_app(repository=repo, analysis_service=AnalysisService(repository=repo, runtime=runtime, frame_service=frame_service))
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    response = analyze(media.id, AnalyzeRequest(duration_ms=2100))
    media_detail = detail(media.id)

    assert response["segment_count"] == 3
    assert response["frame_manifest"]["status"] == "ready"
    assert media_detail["frame_manifest"]["status"] == "ready"
    assert media_detail["warning"]["decision"] == "vlm_required"


def test_analysis_api_uses_stored_media_duration_when_request_omits_duration(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media = _media(tmp_path).model_copy(update={"duration_ms": 2500})
    repo.upsert_media(media)
    predictor = FakePredictor()
    runtime = TribeRuntime(predictor=predictor, npz_dir=tmp_path / "npz")
    frame_service = PlotBrainFrameService(renderer=FakeRenderer(), frames_dir=tmp_path / "frames")
    app = create_app(repository=repo, analysis_service=AnalysisService(repository=repo, runtime=runtime, frame_service=frame_service))
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})

    response = analyze(media.id, AnalyzeRequest())

    assert response["segment_count"] == 3
    assert [(w.start_ms, w.end_ms) for w in predictor.windows] == [
        (0, 1000),
        (1000, 2000),
        (2000, 2500),
    ]


def test_analysis_api_requires_duration_when_media_duration_is_unknown(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media = _media(tmp_path).model_copy(update={"duration_ms": None})
    repo.upsert_media(media)
    runtime = TribeRuntime(predictor=FakePredictor(), npz_dir=tmp_path / "npz")
    frame_service = PlotBrainFrameService(renderer=FakeRenderer(), frames_dir=tmp_path / "frames")
    app = create_app(repository=repo, analysis_service=AnalysisService(repository=repo, runtime=runtime, frame_service=frame_service))
    analyze = _endpoint(app, "/api/media/{media_id}/analyze", {"POST"})

    try:
        analyze(media.id, AnalyzeRequest())
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
        assert "duration_ms is required" in str(exc)
    else:
        raise AssertionError("unknown media duration must fail clearly without request duration")


def _media(tmp_path: Path):
    media_dir = tmp_path / "media"
    media_dir.mkdir(exist_ok=True)
    (media_dir / "story.txt").write_text("local text", encoding="utf-8")
    return scan_media_folder(media_dir)[0]


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
