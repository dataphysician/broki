from pathlib import Path

import brainrot_guard.__main__ as cli
from brainrot_guard.demo_validation import validate_demo_run
from brainrot_guard.repository import Repository


def test_demo_validation_proves_offline_mixed_media_poc(tmp_path: Path) -> None:
    media_dir = _mixed_media_dir(tmp_path)
    repo = Repository(tmp_path / "state.sqlite3")

    report = validate_demo_run(
        repository=repo,
        media_dir=media_dir,
        artifacts_dir=tmp_path / "artifacts",
        duration_ms=3000,
    )

    assert report["ready"] is True
    assert report["media_count"] == 4
    assert report["artifact_integrity"] == "ready"
    assert report["review_ready"] is True
    assert report["privacy_ready"] is True
    assert report["learning_ready"] is True
    assert report["browser_requested"] is False
    assert report["browser_status"] == "not_requested"
    assert {item["frame_manifest_status"] for item in report["artifacts"]} == {"ready"}
    assert {item["engagement_gate_crossed"] for item in report["artifacts"]} == {True}
    assert {item["risk_gate_crossed"] for item in report["artifacts"]} == {True}
    assert {item["warning_decision"] for item in report["artifacts"]} == {"warning_ready"}
    assert {item["vlm_status"] for item in report["artifacts"]} == {"complete"}
    assert report["gate_summary"] == {
        "engagement_gate_crossed_count": 4,
        "risk_gate_crossed_count": 4,
        "warning_ready_count": 4,
        "vlm_required_count": 0,
        "allow_count": 0,
    }
    assert {repo.get_vlm_status(item.id)["status"] for item in repo.list_media()} == {"complete"}


def test_demo_validation_can_include_browser_review_proof(tmp_path: Path) -> None:
    media_dir = _mixed_media_dir(tmp_path)
    repo = Repository(tmp_path / "state.sqlite3")
    seen = {}

    def runner(app, *, media_id: str, screenshot_path: Path | None):
        seen["media_id"] = media_id
        seen["title"] = app.title
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

    report = validate_demo_run(
        repository=repo,
        media_dir=media_dir,
        artifacts_dir=tmp_path / "artifacts",
        include_browser=True,
        browser_screenshot_path=tmp_path / "review.png",
        browser_runner=runner,
    )

    assert report["ready"] is True
    assert report["browser_requested"] is True
    assert report["browser_ready"] is True
    assert report["browser_status"] == "ready"
    assert report["browser"]["screenshot_bytes"] > 0
    assert seen["title"] == "Brainrot Guard POC"
    assert seen["media_id"] == report["browser"]["media_id"]


def test_demo_validation_rejects_youtube_placeholder(tmp_path: Path) -> None:
    media_dir = _mixed_media_dir(tmp_path)
    (media_dir / "youtube.md").write_text("https://youtube.com/watch?v=abc", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")

    report = validate_demo_run(
        repository=repo,
        media_dir=media_dir,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert report["ready"] is False
    assert report["privacy_ready"] is False
    assert report["forbidden_source_count"] == 1
    assert report["browser_requested"] is False
    assert "YouTube" in report["message"]


def test_cli_validate_demo_prints_json(tmp_path: Path, capsys) -> None:
    media_dir = _mixed_media_dir(tmp_path)

    assert cli.main(
        [
            "validate-demo",
            "--media-dir",
            str(media_dir),
            "--db-path",
            str(tmp_path / "state.sqlite3"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ]
    ) == 0
    report = __import__("json").loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["media_count"] == 4


def test_cli_validate_demo_accepts_browser_options(monkeypatch, tmp_path: Path, capsys) -> None:
    media_dir = _mixed_media_dir(tmp_path)

    def fake_validate_demo_run(**kwargs):
        assert kwargs["media_dir"] == media_dir
        assert kwargs["include_browser"] is True
        assert kwargs["browser_screenshot_path"] == tmp_path / "review.png"
        return {"ready": True, "browser_requested": True, "browser_status": "ready", "message": "ready"}

    monkeypatch.setattr(cli, "validate_demo_run", fake_validate_demo_run)

    assert (
        cli.main(
            [
                "validate-demo",
                "--media-dir",
                str(media_dir),
                "--db-path",
                str(tmp_path / "state.sqlite3"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--include-browser",
                "--browser-screenshot-path",
                str(tmp_path / "review.png"),
            ]
        )
        == 0
    )
    report = __import__("json").loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["browser_requested"] is True


def _mixed_media_dir(tmp_path: Path) -> Path:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("Counting by twos\n2 4 6 8\n", encoding="utf-8")
    (media_dir / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (media_dir / "sound.mp3").write_bytes(b"ID3")
    (media_dir / "clip.mp4").write_bytes(b"demo video")
    return media_dir
