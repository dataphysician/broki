from __future__ import annotations

from typing import Any

from brainrot_guard.models import SegmentSignal


def frame_manifest(media_id: str, segments: list[SegmentSignal]) -> dict[str, Any]:
    broken_frame_count = sum(1 for s in segments if s.frame_path is not None and not s.frame_path.exists())
    missing_frame_count = sum(1 for s in segments if s.frame_path is None)
    frames = [
        {
            "timestep": s.timestep,
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "url": f"/api/media/{media_id}/frames/{s.timestep:06d}.png",
        }
        for s in segments
        if s.frame_path is not None and s.frame_path.exists()
    ]
    if broken_frame_count:
        status = "error"
        message = "PlotBrain frame artifact is missing or unreadable."
    elif frames and len(frames) == len(segments):
        status = "ready"
        message = "PlotBrain frame artifacts are ready."
    else:
        status = "pending"
        message = "Waiting for PlotBrain frame artifacts."
    return {
        "status": status,
        "label": "TRIBE-derived neural response proxy",
        "message": message,
        "missing_frame_count": missing_frame_count,
        "broken_frame_count": broken_frame_count,
        "frames": frames,
    }
