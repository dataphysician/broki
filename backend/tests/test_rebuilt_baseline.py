from pathlib import Path

import numpy as np

from brainrot_guard.app import create_app
from brainrot_guard.artifacts import write_segment_npz
from brainrot_guard.demo import generate_demo
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal
from brainrot_guard.repository import Repository


def test_ingestion_rejects_youtube_text_and_accepts_mixed_local_files(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "story.txt").write_text("counting by twos", encoding="utf-8")
    (media_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (media_dir / "link.md").write_text("https://youtube.com/watch?v=abc", encoding="utf-8")

    items = scan_media_folder(media_dir)

    assert {item.kind.value for item in items} == {"text", "image"}
    assert all(item.source == "local_folder" for item in items)


def test_segment_npz_contract_requires_fsaverage5_vertices(tmp_path: Path) -> None:
    path = write_segment_npz(
        tmp_path,
        media_id="media1",
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )

    with np.load(path) as data:
        assert int(data["timestep"]) == 0
        assert int(data["start_ms"]) == 0
        assert int(data["end_ms"]) == 1000
        assert str(data["mesh"]) == "fsaverage5"
        assert data["vertex_values"].shape == (20484,)


def test_review_api_exposes_proxy_label_frames_and_two_gate_warning(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "story.txt").write_text("demo story", encoding="utf-8")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = generate_demo(repo, media_dir, tmp_path / "artifacts", duration_ms=2100)
    app = create_app(repository=repo)
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    response = detail(media_id)

    assert response["label"] == "TRIBE-derived neural response proxy"
    assert "not actual brain activity" in response["caveat"]
    assert response["frame_manifest"]["status"] == "ready"
    assert response["frame_manifest"]["frames"][0]["url"].endswith("/frames/000000.png")
    assert response["warning"]["decision"] == "warning_ready"


def test_demo_generates_segment_frames_for_mixed_local_folder(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "story.txt").write_text("demo story", encoding="utf-8")
    (media_dir / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (media_dir / "clip.mp4").write_bytes(b"demo video")
    (media_dir / "sound.mp3").write_bytes(b"ID3")
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()

    selected_id = generate_demo(repo, media_dir, tmp_path / "artifacts", duration_ms=2100)

    media = repo.list_media()
    assert selected_id in {item.id for item in media}
    assert {item.kind.value for item in media} == {"text", "image", "video", "audio"}
    for item in media:
        segments = repo.list_segments(item.id)
        assert len(segments) == 3
        assert all(segment.npz_path.exists() for segment in segments)
        assert all(segment.frame_path and segment.frame_path.exists() for segment in segments)
        assert repo.get_vlm(item.id).theme == "demo local media"


def test_warning_requires_vlm_after_high_engagement(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_path = media_dir / "story.txt"
    media_path.write_text("demo story", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    repo.upsert_media(media)
    npz_path = write_segment_npz(
        tmp_path / "npz",
        media_id=media.id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    repo.upsert_segments(
        media.id,
        [
            SegmentSignal(
                timestep=0,
                start_ms=0,
                end_ms=1000,
                attention=0.7,
                engagement=0.95,
                arousal=0.8,
                confidence=0.9,
                npz_path=npz_path,
            )
        ],
    )
    app = create_app(repository=repo)
    detail = _endpoint(app, "/api/media/{media_id}", {"GET"})

    response = detail(media.id)

    assert response["warning"]["decision"] == "vlm_required"
    assert response["vlm"] is None


def test_static_review_console_advances_image_and_text_timeline() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")
    html = _static_path("review.html").read_text(encoding="utf-8")

    assert "startStaticPlaybackClock" in script
    assert "setInterval" in script
    assert 'media.kind === "image"' in script
    assert 'media.kind === "video"' in script
    assert "updatePlaybackTime(seconds, duration)" in script
    assert 'id="timeline-playhead"' in html
    assert "Model-predicted engagement/arousal response, not actual brain activity." in html


def test_review_console_renders_vlm_status_and_learned_threshold_terms() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")

    for text in [
        "VLM status",
        "Provider",
        "Provider error",
        "Engagement threshold",
        "Risk threshold",
        "Beta disapproval",
        "GP disapproval",
        "Thompson sample",
        "Threshold source",
        "detail.vlm_status",
        "detail.warning.thresholds",
    ]:
        assert text in script


def test_review_console_can_trigger_analysis_and_show_runtime_errors() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")
    html = _static_path("review.html").read_text(encoding="utf-8")

    assert 'id="analyze-button"' in html
    assert 'id="analysis-state"' in html
    assert "async function runAnalysis" in script
    assert "analysisState.textContent = \"analyzing\"" in script
    assert "analyzeButton.disabled = true" in script
    assert "`/api/media/${currentDetail.media.id}/analyze`" in script
    assert "duration_ms" in script
    assert "analysis error" in script
    assert "await loadMedia(currentDetail.media.id)" in script


def test_review_console_can_opt_into_hiding_skip_recommended_items() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")
    html = _static_path("review.html").read_text(encoding="utf-8")

    assert 'id="skip-recommended-toggle"' in html
    assert "exclude_skip_recommended=true" in script
    assert "skipRecommendedToggle.addEventListener" in script


def test_review_console_can_simulate_skipping_recommended_items() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")
    html = _static_path("review.html").read_text(encoding="utf-8")

    assert 'id="skip-button"' in html
    assert 'id="skip-state"' in html
    assert "skipButton.disabled = decision !== \"skip_recommended\"" in script
    assert "async function skipCurrentRecommended" in script
    assert "skipRecommendedToggle.checked = true" in script
    assert "await loadMediaList(currentDetail.media.id)" in script
    assert "stopAllPlayback()" in script


def test_review_console_clears_stale_media_when_filtered_queue_is_empty() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")

    assert "clearReviewConsole" in script
    assert "currentDetail = null" in script
    assert "No media items match current queue filter." in script
    assert "brainFrame.removeAttribute(\"src\")" in script
    assert "node.removeAttribute(\"src\")" in script


def test_review_console_refreshes_thresholds_after_caregiver_feedback() -> None:
    script = _static_path("review.js").read_text(encoding="utf-8")
    html = _static_path("review.html").read_text(encoding="utf-8")

    assert 'id="feedback-state"' in html
    assert "feedbackState.textContent = \"saving feedback\"" in script
    assert "approveButton.disabled = true" in script
    assert "disapproveButton.disabled = true" in script
    assert "await loadMedia(currentDetail.media.id)" in script
    assert "feedbackState.textContent = `saved ${label}`" in script
    assert "feedback error" in script


def _endpoint(app, path: str, methods: set[str]):
    for route in app.routes:
        if getattr(route, "path", None) == path and getattr(route, "methods", set()) == methods:
            return route.endpoint
    raise AssertionError(f"route not found: {methods} {path}")


def _static_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "brainrot_guard" / "static" / name
