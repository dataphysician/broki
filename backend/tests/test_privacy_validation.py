from pathlib import Path

import pytest

import brainrot_guard.__main__ as cli
from brainrot_guard.privacy_validation import validate_privacy_boundaries
from brainrot_guard.repository import Repository


def test_privacy_validation_passes_for_local_folder_and_schema(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()

    report = validate_privacy_boundaries(repository=repo, media_dir=media_dir)

    assert report["ready"] is True
    assert report["forbidden_source_count"] == 0
    assert report["forbidden_schema_fields"] == []
    assert report["feedback_labels_allowed"] == ["approve", "disapprove"]
    assert report["non_binary_feedback_rejected"] is True


def test_privacy_validation_reports_youtube_placeholders(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "youtube.md").write_text("https://youtu.be/abc", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()

    report = validate_privacy_boundaries(repository=repo, media_dir=media_dir)

    assert report["ready"] is False
    assert report["forbidden_source_count"] == 1
    assert report["forbidden_sources"] == ["youtube.md"]
    assert "YouTube" in report["message"]


def test_privacy_validation_reports_child_telemetry_schema_fields(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    with repo.connect() as conn:
        conn.execute("ALTER TABLE media ADD COLUMN watch_time INTEGER")

    report = validate_privacy_boundaries(repository=repo)

    assert report["ready"] is False
    assert report["forbidden_schema_fields"] == ["media.watch_time"]
    assert "telemetry" in report["message"]


def test_privacy_validation_fails_if_non_binary_feedback_is_accepted(tmp_path: Path) -> None:
    class BadRepository(Repository):
        def record_feedback(self, media_id: str, label: str) -> None:
            return None

        def record_feedback_example(self, media_id: str, label: str, features: tuple[float, ...]) -> bool:
            return True

    repo = BadRepository(tmp_path / "state.sqlite3")
    repo.initialize()

    report = validate_privacy_boundaries(repository=repo)

    assert report["ready"] is False
    assert report["non_binary_feedback_rejected"] is False
    assert "binary" in report["message"]


def test_cli_validate_privacy_prints_json(tmp_path: Path, capsys) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")

    assert cli.main(
        [
            "validate-privacy",
            "--db-path",
            str(tmp_path / "state.sqlite3"),
            "--media-dir",
            str(media_dir),
        ]
    ) == 0
    report = __import__("json").loads(capsys.readouterr().out)

    assert report["ready"] is True
