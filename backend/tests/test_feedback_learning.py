from pathlib import Path

import numpy as np

from brainrot_guard.app import create_app
from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal, VLMDecomposition
from brainrot_guard.repository import Repository


def test_caregiver_disapproval_persists_gp_example_and_recommends_skip_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    repo = Repository(db_path)
    repo.initialize()
    media_a = _register_item(repo, tmp_path, "a.txt", engagement=0.92, risk=0.9)
    app = create_app(repository=repo)
    feedback = _endpoint(app, "/api/media/{media_id}/feedback", {"POST"})

    result = feedback(media_a, _request("disapprove"))

    assert result["feedback"]["label"] == "disapprove"
    assert result["gp_training_example_recorded"] is True
    assert repo.count_feedback_examples() == 1

    restarted = Repository(db_path)
    restarted.initialize()
    media_b = _register_item(restarted, tmp_path, "b.txt", engagement=0.91, risk=0.88)
    restarted_app = create_app(repository=restarted)
    detail = _endpoint(restarted_app, "/api/media/{media_id}", {"GET"})

    response = detail(media_b)

    assert response["warning"]["decision"] == "skip_recommended"
    assert response["warning"]["skip_recommendation"]["reason"] == "similar_to_disapproved_content"
    assert response["warning"]["skip_recommendation"]["probability"] >= 0.8


def test_parent_approval_lowers_similar_skip_probability(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    repo = Repository(db_path)
    repo.initialize()
    media_a = _register_item(repo, tmp_path, "a.txt", engagement=0.92, risk=0.9)
    app = create_app(repository=repo)
    feedback = _endpoint(app, "/api/media/{media_id}/feedback", {"POST"})

    feedback(media_a, _request("approve"))

    restarted = Repository(db_path)
    restarted.initialize()
    media_b = _register_item(restarted, tmp_path, "b.txt", engagement=0.91, risk=0.88)
    detail = _endpoint(create_app(repository=restarted), "/api/media/{media_id}", {"GET"})

    response = detail(media_b)

    assert response["warning"]["decision"] == "warning_ready"
    assert response["warning"]["skip_recommendation"]["probability"] < 0.8


def test_media_list_can_exclude_skip_recommended_items_without_changing_default(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    repo = Repository(db_path)
    repo.initialize()
    media_a = _register_item(repo, tmp_path, "a.txt", engagement=0.92, risk=0.9)
    repo.record_feedback_example(
        media_a,
        "disapprove",
        (0.92, 0.8, 0.7, 0.9, 0.9, 0.85, 0.75, 0.92, 0.88, 0.8, 0.9),
    )
    media_b = _register_item(repo, tmp_path, "b.txt", engagement=0.91, risk=0.88)
    app = create_app(repository=repo)
    media_list = _endpoint(app, "/api/media", {"GET"})

    default_items = media_list()
    filtered_items = media_list(exclude_skip_recommended=True)

    assert {item["id"] for item in default_items} == {media_a, media_b}
    assert media_b not in {item["id"] for item in filtered_items}


def test_feedback_rejects_labels_outside_caregiver_binary_actions(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_item(repo, tmp_path, "a.txt", engagement=0.92, risk=0.9)
    feedback = _endpoint(create_app(repository=repo), "/api/media/{media_id}/feedback", {"POST"})

    try:
        feedback(media_id, _request("watched_long"))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422 or "approve/disapprove" in str(exc)
    else:
        raise AssertionError("non-caregiver behavioral labels must be rejected")


def _register_item(repo: Repository, tmp_path: Path, name: str, *, engagement: float, risk: float) -> str:
    media_dir = tmp_path / f"media-{name}"
    media_dir.mkdir()
    (media_dir / name).write_text("local fixture", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    repo.upsert_media(media)
    npz = write_segment_npz(
        tmp_path / "npz",
        media_id=media.id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32) * engagement,
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
                engagement=engagement,
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
            theme="toy unboxing",
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
    return media.id


class _request:
    def __init__(self, label: str) -> None:
        self.label = label


def _endpoint(app, path: str, methods: set[str]):
    for route in app.routes:
        if getattr(route, "path", None) == path and getattr(route, "methods", set()) == methods:
            return route.endpoint
    raise AssertionError(f"route not found: {methods} {path}")
