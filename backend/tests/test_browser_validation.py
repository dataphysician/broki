from pathlib import Path

import numpy as np

import brainrot_guard.__main__ as cli
from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.browser_validation import validate_browser_review_console
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal, VLMDecomposition
from brainrot_guard.repository import Repository


def test_browser_validation_uses_runner_and_reports_visible_review_console(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_ready_media(repo, tmp_path)

    def runner(app, *, media_id: str, screenshot_path: Path | None):
        assert media_id
        assert app.title == "Brainrot Guard POC"
        screenshot_path.write_bytes(b"\x89PNG\r\n\x1a\nbrowser")
        return {
            "browser_status": "ready",
            "visible_media_stage": True,
            "visible_brain_frame": True,
            "visible_timeline_playhead": True,
            "visible_warning_state": "warning_ready",
            "visible_feedback_controls": True,
            "visible_skip_controls": True,
            "visible_proxy_label": True,
            "screenshot_path": str(screenshot_path),
            "screenshot_bytes": screenshot_path.stat().st_size,
        }

    report = validate_browser_review_console(
        repository=repo,
        media_id=media_id,
        screenshot_path=tmp_path / "browser.png",
        runner=runner,
    )

    assert report["ready"] is True
    assert report["browser_status"] == "ready"
    assert report["visible_warning_state"] == "warning_ready"
    assert report["visible_skip_controls"] is True
    assert report["screenshot_bytes"] > 0


def test_browser_validation_reports_missing_playwright_without_runner(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_ready_media(repo, tmp_path)

    report = validate_browser_review_console(repository=repo, media_id=media_id)

    assert report["ready"] is False
    assert report["browser_status"] == "not_available"
    assert "playwright" in report["message"].lower()


def test_browser_validation_can_bootstrap_demo_artifacts_from_media_folder(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    (media_dir / "poster.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    seen = {}

    def runner(app, *, media_id: str, screenshot_path: Path | None):
        seen["media_id"] = media_id
        screenshot_path.write_bytes(b"\x89PNG\r\n\x1a\nbrowser")
        return {
            "browser_status": "ready",
            "visible_media_stage": True,
            "visible_brain_frame": True,
            "visible_timeline_playhead": True,
            "visible_warning_state": "warning_ready",
            "visible_feedback_controls": True,
            "visible_skip_controls": True,
            "visible_proxy_label": True,
            "screenshot_path": str(screenshot_path),
            "screenshot_bytes": screenshot_path.stat().st_size,
        }

    report = validate_browser_review_console(
        repository=repo,
        media_dir=media_dir,
        artifacts_dir=tmp_path / "artifacts",
        screenshot_path=tmp_path / "browser.png",
        runner=runner,
    )

    assert report["ready"] is True
    assert report["browser_status"] == "ready"
    assert report["bootstrapped_demo"] is True
    assert report["media_count"] == 2
    assert seen["media_id"] == report["media_id"]
    assert repo.list_segments(report["media_id"])


def test_browser_validation_reports_empty_bootstrap_media_folder(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "notes.md").write_text("https://www.youtube.com/watch?v=placeholder", encoding="utf-8")

    report = validate_browser_review_console(
        repository=repo,
        media_dir=media_dir,
        artifacts_dir=tmp_path / "artifacts",
        runner=lambda *args, **kwargs: {"browser_status": "ready"},
    )

    assert report["ready"] is False
    assert report["browser_status"] == "not_checked"
    assert report["bootstrapped_demo"] is False
    assert report["media_count"] == 0
    assert "no supported media" in report["message"]


def test_cli_validate_browser_prints_json_with_injected_runner(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_ready_media(repo, tmp_path)

    def fake_validate_browser_review_console(**kwargs):
        assert kwargs["media_id"] == media_id
        return {"ready": True, "browser_status": "ready", "message": "ready"}

    monkeypatch.setattr(cli, "validate_browser_review_console", fake_validate_browser_review_console)

    assert cli.main(["validate-browser", "--db-path", str(tmp_path / "state.sqlite3"), "--media-id", media_id]) == 0
    report = __import__("json").loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["browser_status"] == "ready"


def test_cli_validate_browser_accepts_artifacts_dir_for_demo_bootstrap(monkeypatch, tmp_path: Path, capsys) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by threes", encoding="utf-8")

    def fake_validate_browser_review_console(**kwargs):
        assert kwargs["media_dir"] == media_dir
        assert kwargs["artifacts_dir"] == tmp_path / "artifacts"
        assert kwargs["duration_ms"] == 2000
        return {"ready": True, "browser_status": "ready", "bootstrapped_demo": True, "message": "ready"}

    monkeypatch.setattr(cli, "validate_browser_review_console", fake_validate_browser_review_console)

    assert (
        cli.main(
            [
                "validate-browser",
                "--db-path",
                str(tmp_path / "state.sqlite3"),
                "--media-dir",
                str(media_dir),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--duration-ms",
                "2000",
            ]
        )
        == 0
    )
    report = __import__("json").loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["bootstrapped_demo"] is True


def _register_ready_media(repo: Repository, tmp_path: Path) -> str:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    repo.upsert_media(media)
    npz = write_segment_npz(
        tmp_path / "npz",
        media_id=media.id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    frame = render_demo_frame(tmp_path / "frames", media_id=media.id, timestep=0)
    repo.upsert_segments(
        media.id,
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
                frame_path=frame,
            )
        ],
    )
    repo.record_vlm(
        media.id,
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
            risk_score=0.91,
            risk_rationale="test fixture",
        ),
    )
    return media.id
