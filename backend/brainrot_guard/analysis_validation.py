from __future__ import annotations

from pathlib import Path
from typing import Any

from brainrot_guard.app_types import frame_manifest
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import AnalysisService, load_npz_prediction


def validate_runtime_analysis(
    *,
    media_dir: Path,
    repository: Repository,
    analysis_service: AnalysisService | None,
    limit: int = 1,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    root = media_dir.expanduser().resolve()
    report: dict[str, Any] = {
        "ready": False,
        "media_dir": str(root),
        "media_count": 0,
        "analyzed_count": 0,
        "results": [],
        "message": "not checked",
    }
    if analysis_service is None:
        report["message"] = "analysis runtime is not configured"
        return report
    if limit <= 0:
        report["message"] = "limit must be positive"
        return report

    try:
        media_items = scan_media_folder(root)
    except Exception as exc:
        report["message"] = str(exc)
        return report

    for item in media_items:
        repository.upsert_media(item)
    report["media_count"] = len(media_items)
    if not media_items:
        report["message"] = "no supported local media files found"
        return report

    engagement_gate_threshold = _analysis_engagement_threshold(analysis_service)
    for item in media_items[:limit]:
        item_duration = duration_ms or item.duration_ms
        if item_duration is None:
            report["results"].append(
                {
                    "media_id": item.id,
                    "filename": item.path.name,
                    "status": "error",
                    "message": "duration_ms is required when media duration is unknown",
                }
            )
            continue
        try:
            analysis_service.analyze(item.id, duration_ms=item_duration)
            result = _artifact_result(
                repository,
                item.id,
                filename=item.path.name,
                engagement_gate_threshold=engagement_gate_threshold,
            )
        except Exception as exc:
            result = {
                "media_id": item.id,
                "filename": item.path.name,
                "status": "error",
                "message": str(exc),
            }
        report["results"].append(result)
        if result.get("status") == "ready":
            report["analyzed_count"] += 1

    errors = [result for result in report["results"] if result.get("status") != "ready"]
    if errors:
        report["message"] = f"{len(errors)} analysis validation item(s) failed"
        return report
    report["ready"] = report["analyzed_count"] > 0
    report["message"] = "ready" if report["ready"] else "no media analyzed"
    return report


def _artifact_result(
    repository: Repository,
    media_id: str,
    *,
    filename: str,
    engagement_gate_threshold: float = 0.8,
) -> dict[str, Any]:
    segments = repository.list_segments(media_id)
    manifest = frame_manifest(media_id, segments)
    npz_count = sum(1 for segment in segments if segment.npz_path.exists())
    frame_count = sum(1 for segment in segments if segment.frame_path is not None and segment.frame_path.exists())
    artifact_failures = _artifact_failures(segments)
    vlm_status = repository.get_vlm_status(media_id)
    vlm = repository.get_vlm(media_id)
    gate_evidence = _gate_evidence(segments, vlm, engagement_gate_threshold=engagement_gate_threshold)
    status = (
        "ready"
        if segments
        and npz_count == len(segments)
        and frame_count == len(segments)
        and not artifact_failures
        else "error"
    )
    result = {
        "media_id": media_id,
        "filename": filename,
        "status": status,
        "segment_count": len(segments),
        "npz_artifact_count": npz_count,
        "frame_artifact_count": frame_count,
        "artifact_integrity": "ready" if not artifact_failures else "error",
        "artifact_failures": artifact_failures,
        "frame_manifest_status": manifest["status"],
        "vlm_status": (vlm_status or {}).get("status"),
        **gate_evidence,
    }
    if status != "ready":
        result["message"] = "analysis did not persist all segment npz and PlotBrain frame artifacts"
    return result


def _analysis_engagement_threshold(analysis_service: AnalysisService) -> float:
    vlm_service = getattr(analysis_service, "vlm_service", None)
    threshold = getattr(vlm_service, "engagement_threshold", None)
    if threshold is None:
        return 0.8
    return float(threshold)


def _gate_evidence(segments, vlm, *, engagement_gate_threshold: float) -> dict[str, Any]:
    max_engagement = max((segment.engagement for segment in segments), default=0.0)
    engagement_gate_crossed = max_engagement >= engagement_gate_threshold
    if not engagement_gate_crossed:
        warning_decision = "allow"
    elif vlm is None:
        warning_decision = "vlm_required"
    else:
        warning_decision = "warning_ready" if vlm.risk_score >= 0.7 else "allow"
    return {
        "max_engagement": round(max_engagement, 6),
        "engagement_gate_threshold": engagement_gate_threshold,
        "engagement_gate_crossed": engagement_gate_crossed,
        "risk_gate_threshold": 0.7,
        "risk_gate_crossed": bool(vlm is not None and vlm.risk_score >= 0.7),
        "warning_decision": warning_decision,
    }


def _artifact_failures(segments) -> list[str]:
    failures: list[str] = []
    for segment in segments:
        try:
            prediction = load_npz_prediction(segment.npz_path)
        except Exception as exc:
            failures.append(f"npz {segment.timestep:06d} failed to load: {exc}")
            continue
        if (
            prediction.timestep != segment.timestep
            or prediction.start_ms != segment.start_ms
            or prediction.end_ms != segment.end_ms
        ):
            failures.append(f"npz {segment.timestep:06d} metadata does not match segment row")
        if segment.frame_path is None:
            failures.append(f"frame {segment.timestep:06d} is missing")
        elif not _is_png(segment.frame_path):
            failures.append(f"frame {segment.timestep:06d} is not a PNG artifact")
    return failures


def _is_png(path: Path) -> bool:
    if not path.exists():
        return False
    return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


__all__ = ["validate_runtime_analysis"]
