from __future__ import annotations

from pathlib import Path
from typing import Any

from brainrot_guard.analysis_validation import _artifact_result
from brainrot_guard.browser_validation import BrowserRunner, validate_browser_review_console
from brainrot_guard.demo import generate_demo_all
from brainrot_guard.learning_validation import validate_learning_calibration
from brainrot_guard.privacy_validation import validate_privacy_boundaries
from brainrot_guard.readiness import validate_local_media_folder
from brainrot_guard.repository import Repository
from brainrot_guard.review_validation import validate_review_console


def validate_demo_run(
    *,
    repository: Repository,
    media_dir: Path,
    artifacts_dir: Path,
    duration_ms: int = 3000,
    include_browser: bool = False,
    browser_screenshot_path: Path | None = None,
    browser_runner: BrowserRunner | None = None,
) -> dict[str, Any]:
    repository.initialize()
    media_report = validate_local_media_folder(media_dir)
    privacy_report = validate_privacy_boundaries(repository=repository, media_dir=media_dir)
    browser_report = _browser_not_requested()
    if media_report["forbidden_source_count"] or not privacy_report["ready"]:
        return {
            "ready": False,
            "media_count": media_report["media_count"],
            "forbidden_source_count": media_report["forbidden_source_count"],
            "privacy_ready": privacy_report["ready"],
            "browser_requested": include_browser,
            "browser_ready": None,
            "browser_status": browser_report["browser_status"],
            "browser": browser_report,
            "message": privacy_report["message"] if not privacy_report["ready"] else media_report["message"],
        }
    if media_report["media_count"] == 0:
        return {
            "ready": False,
            "media_count": 0,
            "forbidden_source_count": 0,
            "privacy_ready": privacy_report["ready"],
            "browser_requested": include_browser,
            "browser_ready": None,
            "browser_status": browser_report["browser_status"],
            "browser": browser_report,
            "message": "no supported local media files found",
        }

    media_ids = generate_demo_all(repository, media_dir, artifacts_dir, duration_ms=duration_ms)
    artifacts = []
    for media_id in media_ids:
        media = repository.get_media(media_id)
        filename = media.path.name if media else media_id
        artifacts.append(_artifact_result(repository, media_id, filename=filename))
    review_report = validate_review_console(repository=repository, limit=len(media_ids))
    learning_report = validate_learning_calibration()
    if include_browser:
        browser_report = validate_browser_review_console(
            repository=repository,
            media_dir=media_dir,
            artifacts_dir=artifacts_dir,
            screenshot_path=browser_screenshot_path,
            duration_ms=duration_ms,
            runner=browser_runner,
        )
    artifact_ready = all(item.get("status") == "ready" for item in artifacts)
    browser_ready = bool(browser_report["ready"]) if include_browser else None
    ready = (
        artifact_ready
        and review_report["ready"]
        and privacy_report["ready"]
        and learning_report["ready"]
        and (not include_browser or bool(browser_ready))
    )
    return {
        "ready": ready,
        "media_count": len(media_ids),
        "media_kinds": media_report["kinds"],
        "forbidden_source_count": media_report["forbidden_source_count"],
        "artifact_integrity": "ready" if artifact_ready else "error",
        "artifacts": artifacts,
        "gate_summary": _gate_summary(artifacts),
        "review_ready": review_report["ready"],
        "privacy_ready": privacy_report["ready"],
        "learning_ready": learning_report["ready"],
        "browser_requested": include_browser,
        "browser_ready": browser_ready,
        "browser_status": browser_report["browser_status"],
        "browser": browser_report,
        "message": "ready" if ready else "offline demo validation failed",
    }


def _gate_summary(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    decisions = [str(item.get("warning_decision")) for item in artifacts]
    return {
        "engagement_gate_crossed_count": sum(1 for item in artifacts if item.get("engagement_gate_crossed") is True),
        "risk_gate_crossed_count": sum(1 for item in artifacts if item.get("risk_gate_crossed") is True),
        "warning_ready_count": decisions.count("warning_ready"),
        "vlm_required_count": decisions.count("vlm_required"),
        "allow_count": decisions.count("allow"),
    }


def _browser_not_requested() -> dict[str, Any]:
    return {
        "ready": True,
        "browser_status": "not_requested",
        "message": "browser validation was not requested",
    }


__all__ = ["validate_demo_run"]
