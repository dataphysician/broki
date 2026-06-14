from __future__ import annotations

from pathlib import Path

from brainrot_guard.learning_validation import _brainrot_like_profile, _educational_profile
from brainrot_guard.readiness_report import validate_readiness_report
from brainrot_guard.repository import Repository


def _stub_analysis_report() -> dict:
    return {
        "ready": True,
        "media_count": 1,
        "analyzed_count": 1,
        "results": [
            {
                "status": "ready",
                "segment_count": 1,
                "npz_artifact_count": 1,
                "frame_artifact_count": 1,
                "frame_provenance_ready_count": 1,
                "scalar_heatmap_bar_count": 0,
                "artifact_integrity": "ready",
                "frame_manifest_status": "ready",
                "engagement_gate_crossed": True,
                "risk_gate_crossed": False,
                "warning_decision": "proceed",
            }
        ],
    }


def _stub_browser_report() -> dict:
    return {
        "ready": True,
        "browser_status": "ready",
        "visible_media_stage": True,
        "visible_brain_frame": True,
        "visible_timeline_playhead": True,
        "playhead_moved": True,
        "brain_frame_synced": True,
        "active_timeline_frame_synced": True,
        "visible_feedback_controls": True,
        "visible_skip_controls": True,
        "visible_auto_close_controls": True,
        "visible_proxy_label": True,
        "screenshot_bytes": 1,
        "missing_visible_checks": [],
    }


def test_readiness_unmet_proofs_disappear_with_stubbed_analysis_and_browser_proofs(
    tmp_path: Path,
) -> None:
    repo = Repository(tmp_path / "test.sqlite3")
    repo.initialize()
    report = validate_readiness_report(
        {},
        repository=repo,
        runtime_analysis_report=_stub_analysis_report(),
        browser_report=_stub_browser_report(),
    )
    assert report["representative_live_analysis"]["ready"] is True
    assert report["browser_visual_proof"]["ready"] is True
    assert "representative_live_analysis" not in report["unmet_proofs"]
    assert "browser_playwright_visual_proof" not in report["external_proof_gaps"]


def test_readiness_unmet_proofs_remain_when_proofs_not_supplied(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "test.sqlite3")
    repo.initialize()
    report = validate_readiness_report({}, repository=repo)
    assert "representative_live_analysis" in report["external_proof_gaps"]
    assert "browser_playwright_visual_proof" in report["external_proof_gaps"]


def test_caregiver_calibration_dataset_proof_passes_with_seeded_balanced_examples(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "test.sqlite3")
    repository.initialize()
    repository.record_feedback_example("media-approve-1", "approve", _educational_profile())
    repository.record_feedback_example("media-approve-2", "approve", _educational_profile())
    repository.record_feedback_example("media-disapprove-1", "disapprove", _brainrot_like_profile())
    repository.record_feedback_example("media-disapprove-2", "disapprove", _brainrot_like_profile())
    repository.record_feedback_example("media-segment-approve", "approve", (0.5, 0.5, 0.5))
    repository.record_feedback_example("media-segment-disapprove", "disapprove", (0.9, 0.9, 0.9))
    report = validate_readiness_report({}, repository=repository)
    assert report["caregiver_calibration_dataset"]["ready"] is True
    assert "broader_caregiver_labeled_calibration" not in report["external_proof_gaps"]
