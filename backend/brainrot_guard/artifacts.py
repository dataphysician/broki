from __future__ import annotations

from pathlib import Path

import numpy as np


def write_segment_npz(base_dir: Path, *, media_id: str, timestep: int, start_ms: int, end_ms: int, vertex_values) -> Path:
    values = np.asarray(vertex_values, dtype=np.float32)
    if values.shape != (20484,):
        raise ValueError("fsaverage5 prediction vectors must contain 20484 vertices")
    path = base_dir / media_id / f"{timestep:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, timestep=timestep, start_ms=start_ms, end_ms=end_ms, mesh="fsaverage5", vertex_values=values)
    return path


def render_demo_frame(base_dir: Path, *, media_id: str, timestep: int) -> Path:
    path = base_dir / media_id / f"{timestep:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1x1 PNG. Tests/demo store it as a rendered frame artifact, never a scalar bar.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
        b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return path
