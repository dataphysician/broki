from __future__ import annotations

from pathlib import Path

import numpy as np

from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal, VLMDecomposition
from brainrot_guard.repository import Repository


def generate_demo(repository: Repository, media_dir: Path, artifacts_dir: Path, duration_ms: int = 3000) -> str:
    media_ids = generate_demo_all(repository, media_dir, artifacts_dir, duration_ms=duration_ms)
    return media_ids[0]


def generate_demo_all(repository: Repository, media_dir: Path, artifacts_dir: Path, duration_ms: int = 3000) -> list[str]:
    items = scan_media_folder(media_dir)
    if not items:
        raise ValueError("no supported media found")
    media_ids = []
    for media in items:
        _generate_item_demo(repository, media, artifacts_dir, duration_ms)
        media_ids.append(media.id)
    return media_ids


def _generate_item_demo(repository: Repository, media, artifacts_dir: Path, duration_ms: int) -> None:
    repository.upsert_media(media)
    segments: list[SegmentSignal] = []
    for timestep, start_ms in enumerate(range(0, duration_ms, 1000)):
        end_ms = min(start_ms + 1000, duration_ms)
        values = np.roll(np.linspace(0, 1, 20484, dtype=np.float32), timestep)
        npz = write_segment_npz(artifacts_dir / "npz", media_id=media.id, timestep=timestep, start_ms=start_ms, end_ms=end_ms, vertex_values=values)
        frame = render_demo_frame(artifacts_dir / "frames", media_id=media.id, timestep=timestep)
        segments.append(SegmentSignal(timestep=timestep, start_ms=start_ms, end_ms=end_ms, attention=0.72, engagement=0.91, arousal=0.76, confidence=0.88, npz_path=npz, frame_path=frame))
    repository.upsert_segments(media.id, segments)
    repository.record_vlm(
        media.id,
        VLMDecomposition(
            theme="demo local media",
            pacing_score=0.9,
            scene_change_cadence_hz=0.8,
            contrast_score=0.8,
            sound_effect_density=0.7,
            educational_value=0.1,
            emotional_hook_score=0.8,
            novelty_score=0.8,
            repetition_score=0.9,
            risk_score=0.9,
            risk_rationale="deterministic demo decomposition for local verification",
        ),
    )
    repository.record_vlm_status(media.id, status="complete", provider="deterministic_demo")
