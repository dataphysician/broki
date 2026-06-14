from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from brainrot_guard.app_types import frame_manifest
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.learning import feature_vector, learned_thresholds, recommend_skip
from brainrot_guard.readiness import (
    validate_hardware_target,
    validate_local_media_folder,
    validate_local_tools,
    validate_tribe_plotbrain,
    validate_vlm_provider,
)
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import AnalysisService

TRIBE_PROXY_LABEL = "TRIBE-derived neural response proxy"
TRIBE_PROXY_CAVEAT = "Model-predicted engagement/arousal response, not actual brain activity."


class FeedbackRequest(BaseModel):
    label: str

    @field_validator("label")
    @classmethod
    def require_binary(cls, value: str) -> str:
        if value not in {"approve", "disapprove"}:
            raise ValueError("feedback label must be approve/disapprove")
        return value


class AnalyzeRequest(BaseModel):
    duration_ms: int | None = Field(default=None, gt=0)


def create_app(
    *,
    repository: Repository,
    media_dir: Path | None = None,
    analysis_service: AnalysisService | None = None,
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    env = environ or os.environ
    repository.initialize()
    if media_dir is not None:
        for item in scan_media_folder(media_dir):
            repository.upsert_media(item)
    app = FastAPI(title="Brainrot Guard POC")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def index():
        return FileResponse(_static("review.html"), media_type="text/html")

    @app.get("/media/{media_id}")
    def media_page(media_id: str):
        if repository.get_media(media_id) is None:
            raise HTTPException(404, "media not found")
        return FileResponse(_static("review.html"), media_type="text/html")

    @app.get("/static/{asset}")
    def static_asset(asset: str):
        path = _static(asset)
        if path.name != asset or not path.exists():
            raise HTTPException(404, "asset not found")
        media_type = "text/css" if path.suffix == ".css" else "application/javascript"
        return FileResponse(path, media_type=media_type)

    @app.get("/api/media")
    def media_list(exclude_skip_recommended: bool = False) -> list[dict[str, Any]]:
        items = repository.list_media()
        if exclude_skip_recommended:
            items = [
                item
                for item in items
                if _warning(
                    repository,
                    repository.list_segments(item.id),
                    repository.get_vlm(item.id),
                )["decision"]
                != "skip_recommended"
            ]
        return [m.model_dump(mode="json") for m in items]

    @app.get("/api/config/tribe/live")
    def tribe_live_config() -> dict[str, Any]:
        return validate_tribe_plotbrain(env)

    @app.get("/api/config/vlm/live")
    def vlm_live_config() -> dict[str, Any]:
        return validate_vlm_provider(env)

    @app.get("/api/config/local/live")
    def local_live_config() -> dict[str, Any]:
        return validate_local_tools(env)

    @app.get("/api/config/hardware/local")
    def local_hardware_config() -> dict[str, Any]:
        return validate_hardware_target(env)

    @app.get("/api/config/media/local")
    def local_media_config() -> dict[str, Any]:
        if media_dir is None:
            return {
                "ready": False,
                "configured": False,
                "media_dir": None,
                "message": "media_dir is not configured",
            }
        report = validate_local_media_folder(media_dir)
        report["configured"] = True
        return report

    @app.get("/api/media/{media_id}")
    def media_detail(media_id: str) -> dict[str, Any]:
        media = repository.get_media(media_id)
        if media is None:
            raise HTTPException(404, "media not found")
        segments = repository.list_segments(media_id)
        vlm = repository.get_vlm(media_id)
        vlm_status = repository.get_vlm_status(media_id)
        warning = _warning(repository, segments, vlm)
        return {
            "media": {**media.model_dump(mode="json"), "file_url": f"/api/media/{media_id}/file"},
            "label": TRIBE_PROXY_LABEL,
            "caveat": TRIBE_PROXY_CAVEAT,
            "segments": [s.model_dump(mode="json") for s in segments],
            "frame_manifest": frame_manifest(media_id, segments),
            "vlm": vlm.model_dump(mode="json") if vlm else None,
            "vlm_status": vlm_status,
            "warning": warning,
        }

    @app.get("/api/media/{media_id}/file")
    def media_file(media_id: str):
        media = repository.get_media(media_id)
        if media is None or not media.path.exists():
            raise HTTPException(404, "media file not found")
        return FileResponse(media.path, media_type=media.mime_type)

    @app.get("/api/media/{media_id}/frames/{frame_name}")
    def frame(media_id: str, frame_name: str):
        segment = repository.get_segment_by_frame(media_id, frame_name)
        if segment is None or segment.frame_path is None or not segment.frame_path.exists():
            raise HTTPException(404, "frame not found")
        return FileResponse(segment.frame_path, media_type="image/png")

    @app.post("/api/media/{media_id}/feedback")
    def feedback(media_id: str, request: FeedbackRequest):
        if repository.get_media(media_id) is None:
            raise HTTPException(404, "media not found")
        repository.record_feedback(media_id, request.label)
        segments = repository.list_segments(media_id)
        vlm = repository.get_vlm(media_id)
        recorded = False
        if segments and vlm is not None:
            recorded = repository.record_feedback_example(media_id, request.label, feature_vector(segments, vlm))
        return {
            "feedback": {"media_id": media_id, "label": request.label},
            "gp_training_example_recorded": recorded,
        }

    @app.post("/api/media/{media_id}/analyze")
    def analyze(media_id: str, request: AnalyzeRequest) -> dict[str, Any]:
        media = repository.get_media(media_id)
        if media is None:
            raise HTTPException(404, "media not found")
        if analysis_service is None:
            raise HTTPException(503, "analysis runtime is not configured")
        duration_ms = request.duration_ms or media.duration_ms
        if duration_ms is None:
            raise HTTPException(400, "duration_ms is required when media duration is unknown")
        return analysis_service.analyze(media_id, duration_ms=duration_ms)

    return app


def _warning(repository: Repository, segments, vlm) -> dict[str, Any]:
    thresholds = None
    if vlm is not None and segments:
        thresholds = learned_thresholds(
            feature_vector(segments, vlm),
            repository.list_feedback_examples(),
        )
    decision = _decision(segments, vlm, thresholds=thresholds)
    skip = None
    if vlm is not None and segments:
        skip = recommend_skip(feature_vector(segments, vlm), repository.list_feedback_examples())
        if decision == "warning_ready" and skip.should_skip:
            decision = "skip_recommended"
    return {
        "decision": decision,
        "skip_recommendation": skip.as_dict() if skip else None,
        "thresholds": thresholds or learned_thresholds((), []),
    }


def _decision(segments, vlm, *, thresholds: dict[str, Any] | None = None) -> str:
    engagement_threshold = float((thresholds or {}).get("engagement", 0.8))
    risk_threshold = float((thresholds or {}).get("risk", 0.7))
    high_engagement = any(s.engagement >= engagement_threshold for s in segments)
    if not high_engagement:
        return "allow"
    if vlm is None:
        return "vlm_required"
    return "warning_ready" if vlm.risk_score >= risk_threshold else "allow"


def _static(name: str) -> Path:
    return Path(__file__).resolve().parent / "static" / name
