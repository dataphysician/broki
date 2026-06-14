from pathlib import Path

import numpy as np

import brainrot_guard.__main__ as cli
from brainrot_guard.analysis_validation import validate_runtime_analysis
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import (
    AnalysisService,
    NpzPrediction,
    PlotBrainFrameService,
    SegmentWindow,
    TribePrediction,
    TribeRuntime,
)
from brainrot_guard.vlm import GatedVLMService, VLMDecompositionRequest
from brainrot_guard.models import VLMDecomposition


class FakePredictor:
    def __init__(self, engagement: float = 0.91) -> None:
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


class FakeDecomposer:
    def __init__(self) -> None:
        self.calls: list[VLMDecompositionRequest] = []

    def decompose(self, request: VLMDecompositionRequest) -> VLMDecomposition:
        self.calls.append(request)
        return VLMDecomposition(
            theme="fast reward loop",
            pacing_score=0.9,
            scene_change_cadence_hz=0.8,
            contrast_score=0.85,
            sound_effect_density=0.75,
            educational_value=0.08,
            emotional_hook_score=0.88,
            novelty_score=0.8,
            repetition_score=0.9,
            risk_score=0.91,
            risk_rationale="test fixture",
        )


def test_runtime_analysis_validation_requires_configured_service(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()

    report = validate_runtime_analysis(media_dir=media_dir, repository=repo, analysis_service=None)

    assert report["ready"] is False
    assert report["analyzed_count"] == 0
    assert "analysis runtime is not configured" in report["message"]


def test_runtime_analysis_validation_checks_persisted_npz_and_frames(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    service = _analysis_service(tmp_path, repo)

    report = validate_runtime_analysis(
        media_dir=media_dir,
        repository=repo,
        analysis_service=service,
        limit=1,
    )

    assert report["ready"] is True
    assert report["media_count"] == 1
    assert report["analyzed_count"] == 1
    result = report["results"][0]
    assert result["segment_count"] == 3
    assert result["npz_artifact_count"] == 3
    assert result["frame_artifact_count"] == 3
    assert result["artifact_integrity"] == "ready"
    assert result["frame_manifest_status"] == "ready"
    assert result["vlm_status"] == "not_configured"
    assert result["engagement_gate_crossed"] is True
    assert result["warning_decision"] == "vlm_required"


def test_runtime_analysis_validation_reports_skipped_vlm_gate(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    decomposer = FakeDecomposer()
    service = _analysis_service(tmp_path, repo, engagement=0.3, decomposer=decomposer)

    report = validate_runtime_analysis(
        media_dir=media_dir,
        repository=repo,
        analysis_service=service,
        limit=1,
    )

    result = report["results"][0]
    assert report["ready"] is True
    assert decomposer.calls == []
    assert result["max_engagement"] == 0.3
    assert result["engagement_gate_crossed"] is False
    assert result["vlm_status"] == "skipped_engagement_gate"
    assert result["warning_decision"] == "allow"


def test_runtime_analysis_validation_reports_two_gate_warning(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    decomposer = FakeDecomposer()
    service = _analysis_service(tmp_path, repo, engagement=0.91, decomposer=decomposer)

    report = validate_runtime_analysis(
        media_dir=media_dir,
        repository=repo,
        analysis_service=service,
        limit=1,
    )

    result = report["results"][0]
    assert report["ready"] is True
    assert len(decomposer.calls) == 1
    assert result["max_engagement"] == 0.91
    assert result["engagement_gate_crossed"] is True
    assert result["vlm_status"] == "complete"
    assert result["warning_decision"] == "warning_ready"


def test_runtime_analysis_validation_uses_configured_vlm_engagement_threshold(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    decomposer = FakeDecomposer()
    service = _analysis_service(
        tmp_path,
        repo,
        engagement=0.91,
        decomposer=decomposer,
        engagement_threshold=0.95,
    )

    report = validate_runtime_analysis(
        media_dir=media_dir,
        repository=repo,
        analysis_service=service,
        limit=1,
    )

    result = report["results"][0]
    assert report["ready"] is True
    assert decomposer.calls == []
    assert result["max_engagement"] == 0.91
    assert result["engagement_gate_threshold"] == 0.95
    assert result["engagement_gate_crossed"] is False
    assert result["vlm_status"] == "skipped_engagement_gate"
    assert result["warning_decision"] == "allow"


def test_runtime_analysis_validation_reports_npz_metadata_mismatch(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    service = _analysis_service(tmp_path, repo)
    validate_runtime_analysis(
        media_dir=media_dir,
        repository=repo,
        analysis_service=service,
        limit=1,
    )
    media_id = repo.list_media()[0].id
    segment = repo.list_segments(media_id)[0]
    np.savez_compressed(
        segment.npz_path,
        timestep=segment.timestep,
        start_ms=999,
        end_ms=segment.end_ms,
        mesh="fsaverage5",
        vertex_values=np.ones(20484, dtype=np.float32),
    )

    result = validate_runtime_analysis_artifacts_for_test(repo, media_id)

    assert result["status"] == "error"
    assert result["artifact_integrity"] == "error"
    assert "metadata" in result["artifact_failures"][0]


def test_runtime_analysis_validation_reports_unknown_duration(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "sound.mp3").write_bytes(b"not a real mp3")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()

    report = validate_runtime_analysis(
        media_dir=media_dir,
        repository=repo,
        analysis_service=_analysis_service(tmp_path, repo),
    )

    assert report["ready"] is False
    assert report["analyzed_count"] == 0
    assert report["results"][0]["status"] == "error"
    assert "duration_ms is required" in report["results"][0]["message"]


def test_cli_validate_analysis_prints_json(monkeypatch, tmp_path: Path, capsys) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")

    def fake_build_analysis_service(repository, env):
        return _analysis_service(tmp_path, repository)

    monkeypatch.setattr(cli, "build_analysis_service_from_env", fake_build_analysis_service)

    result = cli.main(
        [
            "validate-analysis",
            "--media-dir",
            str(media_dir),
            "--db-path",
            str(tmp_path / "state.sqlite3"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--enable-live-analysis",
        ]
    )

    assert result == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["ready"] is True
    assert report["analyzed_count"] == 1


def _analysis_service(
    tmp_path: Path,
    repo: Repository,
    *,
    engagement: float = 0.91,
    decomposer: FakeDecomposer | None = None,
    engagement_threshold: float = 0.8,
) -> AnalysisService:
    return AnalysisService(
        repository=repo,
        runtime=TribeRuntime(predictor=FakePredictor(engagement=engagement), npz_dir=tmp_path / "npz"),
        frame_service=PlotBrainFrameService(renderer=FakeRenderer(), frames_dir=tmp_path / "frames"),
        vlm_service=(
            GatedVLMService(decomposer=decomposer, engagement_threshold=engagement_threshold)
            if decomposer is not None
            else None
        ),
    )


def validate_runtime_analysis_artifacts_for_test(repo: Repository, media_id: str):
    from brainrot_guard.analysis_validation import _artifact_result

    return _artifact_result(repo, media_id, filename="lesson.txt")


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)
