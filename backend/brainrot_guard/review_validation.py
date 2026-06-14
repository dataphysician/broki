from __future__ import annotations

from pathlib import Path
from typing import Any

from brainrot_guard.app import TRIBE_PROXY_CAVEAT, TRIBE_PROXY_LABEL, create_app
from brainrot_guard.repository import Repository


def validate_review_console(
    *,
    repository: Repository,
    media_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ready": False,
        "checked_count": 0,
        "items": [],
        "static_contract": _static_contract(),
        "message": "not checked",
    }
    if limit <= 0:
        report["message"] = "limit must be positive"
        return report
    app = create_app(repository=repository)
    detail_endpoint = _endpoint(app, "/api/media/{media_id}", {"GET"})
    media_ids = [media_id] if media_id else [item.id for item in repository.list_media()[:limit]]
    if not media_ids:
        report["message"] = "no media items found"
        return report

    for current_id in media_ids:
        try:
            detail = detail_endpoint(current_id)
        except Exception as exc:
            report["items"].append(
                {
                    "media_id": current_id,
                    "status": "error",
                    "message": _exception_message(exc),
                }
            )
            continue
        report["items"].append(_validate_detail(detail))

    report["checked_count"] = len(report["items"])
    errors = [item for item in report["items"] if item.get("status") != "ready"]
    static_ready = report["static_contract"]["ready"]
    if errors or not static_ready:
        if len(errors) == 1 and errors[0].get("message"):
            report["message"] = str(errors[0]["message"])
        else:
            report["message"] = f"{len(errors)} review item(s) failed" if errors else "static UI contract failed"
        return report
    report["ready"] = True
    report["message"] = "ready"
    return report


def _validate_detail(detail: dict[str, Any]) -> dict[str, Any]:
    static_contract = _static_contract()
    frame_manifest = detail.get("frame_manifest") or {}
    frames = frame_manifest.get("frames") or []
    segments = detail.get("segments") or []
    warning = detail.get("warning") or {}
    failures = []
    if detail.get("label") != TRIBE_PROXY_LABEL:
        failures.append("missing TRIBE proxy label")
    if "not actual brain activity" not in str(detail.get("caveat", "")):
        failures.append("missing proxy caveat")
    if frame_manifest.get("status") not in {"pending", "ready", "error"}:
        failures.append("invalid frame manifest status")
    if frame_manifest.get("status") == "ready" and len(frames) != len(segments):
        failures.append("ready frame manifest must include every segment")
    for frame in frames:
        if not str(frame.get("url", "")).endswith(f"/frames/{int(frame['timestep']):06d}.png"):
            failures.append("frame URL does not match PlotBrain artifact route")
        if int(frame.get("end_ms", 0)) <= int(frame.get("start_ms", 0)):
            failures.append("frame window is not forward")
    decision = warning.get("decision")
    if decision not in {"allow", "vlm_required", "warning_ready", "skip_recommended"}:
        failures.append("invalid warning decision")
    return {
        "media_id": detail.get("media", {}).get("id"),
        "filename": Path(str(detail.get("media", {}).get("path", ""))).name,
        "status": "error" if failures else "ready",
        "failures": failures,
        "has_proxy_label": detail.get("label") == TRIBE_PROXY_LABEL,
        "has_proxy_caveat": detail.get("caveat") == TRIBE_PROXY_CAVEAT,
        "has_feedback_controls": static_contract["has_feedback_controls"],
        "has_feedback_refresh": static_contract["has_feedback_refresh"],
        "has_skip_controls": static_contract["has_skip_controls"],
        "has_vlm_summary": "vlm_status" in detail and "warning" in detail,
        "frame_manifest_status": frame_manifest.get("status"),
        "frame_count": len(frames),
        "segment_count": len(segments),
        "warning_decision": decision,
    }


def _static_contract() -> dict[str, Any]:
    html = _static_path("review.html").read_text(encoding="utf-8")
    script = _static_path("review.js").read_text(encoding="utf-8")
    required_html = [
        'aria-label="Media playback"',
        'aria-label="Review console"',
        'id="brain-frame"',
        'id="timeline-playhead"',
        'id="approve-button"',
        'id="disapprove-button"',
        'id="feedback-state"',
        'id="skip-button"',
        'id="skip-state"',
        TRIBE_PROXY_LABEL,
        "not actual brain activity",
    ]
    required_script = [
        "renderFrameTimeline",
        "updatePlaybackTime",
        "renderBrainFrame",
        "sendFeedback(\"approve\")",
        "sendFeedback(\"disapprove\")",
        "runAnalysis",
        "detail.warning.thresholds",
        "detail.vlm_status",
        "feedbackState.textContent",
        "await loadMedia(currentDetail.media.id)",
        "skipCurrentRecommended",
        "skipRecommendedToggle.checked = true",
    ]
    missing_html = [token for token in required_html if token not in html]
    missing_script = [token for token in required_script if token not in script]
    return {
        "ready": not missing_html and not missing_script,
        "missing_html": missing_html,
        "missing_script": missing_script,
        "has_feedback_controls": 'id="approve-button"' in html and 'id="disapprove-button"' in html,
        "has_feedback_refresh": 'id="feedback-state"' in html and "feedbackState.textContent" in script,
        "has_skip_controls": 'id="skip-button"' in html and 'id="skip-state"' in html and "skipCurrentRecommended" in script,
    }


def _endpoint(app, path: str, methods: set[str]):
    for route in app.routes:
        if getattr(route, "path", None) == path and getattr(route, "methods", set()) == methods:
            return route.endpoint
    raise AssertionError(f"route not found: {methods} {path}")


def _exception_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    return str(detail or exc)


def _static_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "static" / name


__all__ = ["validate_review_console"]
