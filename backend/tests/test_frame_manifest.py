from pathlib import Path

import numpy as np

from brainrot_guard.app_types import frame_manifest
from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.models import SegmentSignal


def test_frame_manifest_is_pending_before_frame_artifacts_exist(tmp_path: Path) -> None:
    segment = _segment(tmp_path, frame_path=None)

    manifest = frame_manifest("media-1", [segment])

    assert manifest["status"] == "pending"
    assert manifest["missing_frame_count"] == 1
    assert manifest["broken_frame_count"] == 0
    assert "Waiting" in manifest["message"]


def test_frame_manifest_is_ready_when_all_frames_exist(tmp_path: Path) -> None:
    frame = render_demo_frame(tmp_path / "frames", media_id="media-1", timestep=0)
    segment = _segment(tmp_path, frame_path=frame)

    manifest = frame_manifest("media-1", [segment])

    assert manifest["status"] == "ready"
    assert manifest["missing_frame_count"] == 0
    assert manifest["broken_frame_count"] == 0
    assert manifest["frames"][0]["url"].endswith("/frames/000000.png")


def test_frame_manifest_reports_error_for_broken_frame_artifact_path(tmp_path: Path) -> None:
    segment = _segment(tmp_path, frame_path=tmp_path / "frames" / "missing.png")

    manifest = frame_manifest("media-1", [segment])

    assert manifest["status"] == "error"
    assert manifest["missing_frame_count"] == 0
    assert manifest["broken_frame_count"] == 1
    assert "missing or unreadable" in manifest["message"]


def _segment(tmp_path: Path, *, frame_path: Path | None) -> SegmentSignal:
    npz = write_segment_npz(
        tmp_path / "npz",
        media_id="media-1",
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    return SegmentSignal(
        timestep=0,
        start_ms=0,
        end_ms=1000,
        attention=0.7,
        engagement=0.8,
        arousal=0.6,
        confidence=0.9,
        npz_path=npz,
        frame_path=frame_path,
    )
