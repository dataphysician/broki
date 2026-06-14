import json
from pathlib import Path

import numpy as np
import pytest

from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal
from brainrot_guard.vlm import VLMDecompositionRequest
from brainrot_guard.vlm.adapters import VLMAdapterError, build_provider_payload


def test_provider_payload_inlines_media_and_plotbrain_frames_without_local_paths(tmp_path: Path) -> None:
    request = _request(tmp_path)

    payload = build_provider_payload(request)
    serialized = json.dumps(payload, sort_keys=True)

    assert str(tmp_path) not in serialized
    assert payload["media"]["filename"] == "story.txt"
    assert payload["media"]["mime_type"] == "text/plain"
    assert payload["media"]["base64"]
    assert payload["plotbrain_frames"][0]["filename"] == "000000.png"
    assert payload["plotbrain_frames"][0]["mime_type"] == "image/png"
    assert payload["segments"][0]["mesh"] == "fsaverage5"
    assert payload["segments"][0]["npz_artifact"] == "000000.npz"


def test_provider_payload_fails_if_media_file_is_missing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.media.path.unlink()

    with pytest.raises(VLMAdapterError, match="local media file is missing"):
        build_provider_payload(request)


def test_provider_payload_fails_if_plotbrain_frame_is_missing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.frame_paths[0].unlink()

    with pytest.raises(VLMAdapterError, match="PlotBrain frame artifact is missing"):
        build_provider_payload(request)


def _request(tmp_path: Path) -> VLMDecompositionRequest:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "story.txt").write_text("counting fixture", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    npz = write_segment_npz(
        tmp_path / "npz",
        media_id=media.id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    frame = render_demo_frame(tmp_path / "frames", media_id=media.id, timestep=0)
    segment = SegmentSignal(
        timestep=0,
        start_ms=0,
        end_ms=1000,
        attention=0.7,
        engagement=0.91,
        arousal=0.8,
        confidence=0.9,
        npz_path=npz,
        frame_path=frame,
    )
    return VLMDecompositionRequest(media=media, segments=[segment], frame_paths=(frame,))
