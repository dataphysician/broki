from pathlib import Path

import numpy as np

import brainrot_guard.__main__ as cli
from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal, VLMDecomposition
from brainrot_guard.repository import Repository
from brainrot_guard.review_validation import validate_review_console


def test_review_console_validation_accepts_pending_plotbrain_state(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_media(repo, tmp_path, "pending.txt")

    report = validate_review_console(repository=repo, media_id=media_id)

    assert report["ready"] is True
    assert report["checked_count"] == 1
    item = report["items"][0]
    assert item["frame_manifest_status"] == "pending"
    assert item["warning_decision"] == "allow"
    assert item["has_proxy_label"] is True
    assert item["has_feedback_controls"] is True
    assert item["has_feedback_refresh"] is True
    assert item["has_skip_controls"] is True


def test_review_console_validation_checks_ready_frames_and_two_gate_warning(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_media(repo, tmp_path, "risky.txt")
    _record_ready_analysis(repo, tmp_path, media_id, engagement=0.92, risk=0.91)

    report = validate_review_console(repository=repo, media_id=media_id)

    assert report["ready"] is True
    item = report["items"][0]
    assert item["frame_manifest_status"] == "ready"
    assert item["frame_count"] == 1
    assert item["segment_count"] == 1
    assert item["warning_decision"] == "warning_ready"
    assert item["has_vlm_summary"] is True


def test_review_console_validation_accepts_error_plotbrain_state(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_media(repo, tmp_path, "broken-frame.txt")
    _record_broken_frame_analysis(repo, tmp_path, media_id)

    report = validate_review_console(repository=repo, media_id=media_id)

    assert report["ready"] is True
    item = report["items"][0]
    assert item["frame_manifest_status"] == "error"
    assert item["frame_count"] == 0
    assert item["segment_count"] == 1


def test_review_console_validation_reports_missing_media(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()

    report = validate_review_console(repository=repo, media_id="missing")

    assert report["ready"] is False
    assert "media not found" in report["message"]


def test_cli_validate_ui_prints_json(tmp_path: Path, capsys) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    _register_media(repo, tmp_path, "pending.txt")

    assert cli.main(["validate-ui", "--db-path", str(tmp_path / "state.sqlite3")]) == 0
    report = __import__("json").loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["checked_count"] == 1
    assert report["static_contract"]["has_feedback_refresh"] is True
    assert report["static_contract"]["has_skip_controls"] is True


def _register_media(repo: Repository, tmp_path: Path, name: str) -> str:
    media_dir = tmp_path / f"media-{name}"
    media_dir.mkdir()
    (media_dir / name).write_text("local fixture", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    repo.upsert_media(media)
    return media.id


def _record_ready_analysis(repo: Repository, tmp_path: Path, media_id: str, *, engagement: float, risk: float) -> None:
    npz = write_segment_npz(
        tmp_path / "npz",
        media_id=media_id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    frame = render_demo_frame(tmp_path / "frames", media_id=media_id, timestep=0)
    repo.upsert_segments(
        media_id,
        [
            SegmentSignal(
                timestep=0,
                start_ms=0,
                end_ms=1000,
                attention=0.7,
                engagement=engagement,
                arousal=0.8,
                confidence=0.9,
                npz_path=npz,
                frame_path=frame,
            )
        ],
    )
    repo.record_vlm(
        media_id,
        VLMDecomposition(
            theme="fast reward loop",
            pacing_score=0.9,
            scene_change_cadence_hz=0.8,
            contrast_score=0.85,
            sound_effect_density=0.75,
            educational_value=0.08,
            emotional_hook_score=0.88,
            novelty_score=0.8,
            repetition_score=0.9,
            risk_score=risk,
            risk_rationale="test fixture",
        ),
    )


def _record_broken_frame_analysis(repo: Repository, tmp_path: Path, media_id: str) -> None:
    npz = write_segment_npz(
        tmp_path / "npz",
        media_id=media_id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    repo.upsert_segments(
        media_id,
        [
            SegmentSignal(
                timestep=0,
                start_ms=0,
                end_ms=1000,
                attention=0.7,
                engagement=0.92,
                arousal=0.8,
                confidence=0.9,
                npz_path=npz,
                frame_path=tmp_path / "frames" / "missing.png",
            )
        ],
    )
