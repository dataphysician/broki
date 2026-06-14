from pathlib import Path

import numpy as np

from brainrot_guard.app import create_app
from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.learning import FeedbackExample, feature_vector, learned_thresholds
from brainrot_guard.models import SegmentSignal, VLMDecomposition
from brainrot_guard.repository import Repository


def test_disapproval_examples_lower_thresholds_for_similar_content_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    repo = Repository(db_path)
    repo.initialize()
    disapproved_id = _register_item(repo, tmp_path, "bad.txt", engagement=0.92, risk=0.9)
    repo.record_feedback_example(disapproved_id, "disapprove", feature_vector(repo.list_segments(disapproved_id), repo.get_vlm(disapproved_id)))

    restarted = Repository(db_path)
    restarted.initialize()
    borderline_id = _register_item(restarted, tmp_path, "borderline.txt", engagement=0.76, risk=0.68)
    detail = _endpoint(create_app(repository=restarted), "/api/media/{media_id}", {"GET"})

    response = detail(borderline_id)

    assert response["warning"]["decision"] == "warning_ready"
    assert response["warning"]["thresholds"]["engagement"] < 0.8
    assert response["warning"]["thresholds"]["risk"] < 0.7
    assert response["warning"]["thresholds"]["source"] == "caregiver_feedback"


def test_approval_examples_keep_borderline_content_below_warning_threshold(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    repo = Repository(db_path)
    repo.initialize()
    approved_id = _register_item(repo, tmp_path, "good.txt", engagement=0.92, risk=0.9)
    repo.record_feedback_example(approved_id, "approve", feature_vector(repo.list_segments(approved_id), repo.get_vlm(approved_id)))

    restarted = Repository(db_path)
    restarted.initialize()
    borderline_id = _register_item(restarted, tmp_path, "borderline.txt", engagement=0.76, risk=0.68)
    detail = _endpoint(create_app(repository=restarted), "/api/media/{media_id}", {"GET"})

    response = detail(borderline_id)

    assert response["warning"]["decision"] == "allow"
    assert response["warning"]["thresholds"]["engagement"] >= 0.8
    assert response["warning"]["thresholds"]["risk"] >= 0.7


def test_learned_thresholds_report_bayesian_terms_without_behavioral_inputs(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_item(repo, tmp_path, "bad.txt", engagement=0.92, risk=0.9)
    target = feature_vector(repo.list_segments(media_id), repo.get_vlm(media_id))
    repo.record_feedback_example(media_id, "disapprove", target)

    thresholds = learned_thresholds(target, repo.list_feedback_examples(), random_seed=7)
    repeated = learned_thresholds(target, repo.list_feedback_examples(), random_seed=7)

    assert thresholds["beta_disapproval_mean"] > 0.5
    assert thresholds["beta_alpha"] == 2.0
    assert thresholds["beta_beta"] == 1.0
    assert thresholds["gp_disapproval_mean"] > 0.5
    assert thresholds["gp_disapproval_variance"] >= 0
    assert thresholds["thompson_sample"] == repeated["thompson_sample"]
    assert thresholds["thompson_sample"] != thresholds["beta_disapproval_mean"]
    assert "retention" not in thresholds
    assert "scroll" not in thresholds


def test_unseeded_thresholds_are_stable_for_same_caregiver_examples(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    media_id = _register_item(repo, tmp_path, "bad.txt", engagement=0.92, risk=0.9)
    target = feature_vector(repo.list_segments(media_id), repo.get_vlm(media_id))
    repo.record_feedback_example(media_id, "disapprove", target)

    first = learned_thresholds(target, repo.list_feedback_examples())
    second = learned_thresholds(target, repo.list_feedback_examples())

    assert first == second


def test_unseeded_thresholds_are_independent_of_temp_media_ids() -> None:
    target = (0.76, 0.8, 0.7, 0.68, 0.9, 0.85, 0.75, 0.92, 0.88, 0.8, 0.9)
    features = (0.92, 0.8, 0.7, 0.9, 0.9, 0.85, 0.75, 0.92, 0.88, 0.8, 0.9)
    first = learned_thresholds(
        target,
        [FeedbackExample("temp-a", "disapprove", features)],
    )
    second = learned_thresholds(
        target,
        [FeedbackExample("temp-b", "disapprove", features)],
    )

    assert first == second
    assert first["engagement"] < 0.76
    assert first["risk"] < 0.68


def test_gp_posterior_is_similarity_sensitive_for_caregiver_examples(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    repo.initialize()
    disapproved_id = _register_item(repo, tmp_path, "bad.txt", engagement=0.92, risk=0.9)
    approved_id = _register_item(repo, tmp_path, "good.txt", engagement=0.2, risk=0.1)
    repo.record_feedback_example(disapproved_id, "disapprove", feature_vector(repo.list_segments(disapproved_id), repo.get_vlm(disapproved_id)))
    repo.record_feedback_example(approved_id, "approve", feature_vector(repo.list_segments(approved_id), repo.get_vlm(approved_id)))

    similar_target = feature_vector(repo.list_segments(disapproved_id), repo.get_vlm(disapproved_id))
    distant_target = feature_vector(repo.list_segments(approved_id), repo.get_vlm(approved_id))

    similar = learned_thresholds(similar_target, repo.list_feedback_examples(), random_seed=3)
    distant = learned_thresholds(distant_target, repo.list_feedback_examples(), random_seed=3)

    assert similar["gp_disapproval_mean"] > distant["gp_disapproval_mean"]
    assert similar["engagement"] < distant["engagement"]


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


def _endpoint(app, path: str, methods: set[str]):
    for route in app.routes:
        if getattr(route, "path", None) == path and getattr(route, "methods", set()) == methods:
            return route.endpoint
    raise AssertionError(f"route not found: {methods} {path}")
