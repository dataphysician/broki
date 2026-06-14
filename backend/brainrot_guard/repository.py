from __future__ import annotations

from pathlib import Path
import json
import sqlite3

from brainrot_guard.learning import FeedbackExample
from brainrot_guard.models import MediaItem, MediaKind, SegmentSignal, VLMDecomposition


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS media (
                  id TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL,
                  mime_type TEXT NOT NULL, duration_ms INTEGER, source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments (
                  media_id TEXT NOT NULL, timestep INTEGER NOT NULL, start_ms INTEGER NOT NULL,
                  end_ms INTEGER NOT NULL, attention REAL NOT NULL, engagement REAL NOT NULL,
                  arousal REAL NOT NULL, confidence REAL NOT NULL, mesh TEXT NOT NULL,
                  vertex_count INTEGER NOT NULL, npz_path TEXT NOT NULL, frame_path TEXT,
                  PRIMARY KEY(media_id, timestep)
                );
                CREATE TABLE IF NOT EXISTS vlm (
                  media_id TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vlm_status (
                  media_id TEXT PRIMARY KEY, status TEXT NOT NULL, provider TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback (
                  media_id TEXT PRIMARY KEY, label TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_examples (
                  media_id TEXT PRIMARY KEY, label TEXT NOT NULL, features_json TEXT NOT NULL
                );
                """
            )

    def upsert_media(self, item: MediaItem) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO media VALUES (?, ?, ?, ?, ?, ?)",
                (item.id, str(item.path), item.kind.value, item.mime_type, item.duration_ms, item.source),
            )

    def list_media(self) -> list[MediaItem]:
        with self.connect() as conn:
            return [_media(row) for row in conn.execute("SELECT * FROM media ORDER BY path")]

    def get_media(self, media_id: str) -> MediaItem | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
            return _media(row) if row else None

    def upsert_segments(self, media_id: str, segments: list[SegmentSignal]) -> None:
        with self.connect() as conn:
            for s in segments:
                conn.execute(
                    "INSERT OR REPLACE INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        media_id,
                        s.timestep,
                        s.start_ms,
                        s.end_ms,
                        s.attention,
                        s.engagement,
                        s.arousal,
                        s.confidence,
                        s.mesh,
                        s.vertex_count,
                        str(s.npz_path),
                        str(s.frame_path) if s.frame_path else None,
                    ),
                )

    def list_segments(self, media_id: str) -> list[SegmentSignal]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM segments WHERE media_id = ? ORDER BY timestep", (media_id,))
            return [_segment(row) for row in rows]

    def get_segment_by_frame(self, media_id: str, frame_name: str) -> SegmentSignal | None:
        timestep = int(Path(frame_name).stem)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM segments WHERE media_id = ? AND timestep = ?",
                (media_id, timestep),
            ).fetchone()
            return _segment(row) if row else None

    def record_vlm(self, media_id: str, decomposition: VLMDecomposition) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO vlm VALUES (?, ?)", (media_id, decomposition.model_dump_json()))

    def get_vlm(self, media_id: str) -> VLMDecomposition | None:
        with self.connect() as conn:
            row = conn.execute("SELECT payload FROM vlm WHERE media_id = ?", (media_id,)).fetchone()
            return VLMDecomposition.model_validate_json(row["payload"]) if row else None

    def record_vlm_status(self, media_id: str, *, status: str, provider: str | None = None, error: str | None = None) -> None:
        if status not in {"not_configured", "skipped_engagement_gate", "complete", "error"}:
            raise ValueError("VLM status must be not_configured, skipped_engagement_gate, complete, or error")
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vlm_status VALUES (?, ?, ?, ?)",
                (media_id, status, provider, error),
            )

    def get_vlm_status(self, media_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM vlm_status WHERE media_id = ?", (media_id,)).fetchone()
            if row is None:
                return None
            return {
                "media_id": media_id,
                "status": row["status"],
                "provider": row["provider"],
                "error": row["error"],
            }

    def record_feedback(self, media_id: str, label: str) -> None:
        if label not in {"approve", "disapprove"}:
            raise ValueError("feedback label must be approve/disapprove")
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?)", (media_id, label))

    def record_feedback_example(self, media_id: str, label: str, features: tuple[float, ...]) -> bool:
        if label not in {"approve", "disapprove"}:
            raise ValueError("feedback label must be approve/disapprove")
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feedback_examples VALUES (?, ?, ?)",
                (media_id, label, json.dumps(list(features))),
            )
        return True

    def list_feedback_examples(self) -> list[FeedbackExample]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM feedback_examples ORDER BY media_id").fetchall()
            return [
                FeedbackExample(
                    media_id=row["media_id"],
                    label=row["label"],
                    features=tuple(float(value) for value in json.loads(row["features_json"])),
                )
                for row in rows
            ]

    def count_feedback_examples(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM feedback_examples").fetchone()
            return int(row["count"])


def _media(row: sqlite3.Row) -> MediaItem:
    return MediaItem(
        id=row["id"],
        path=Path(row["path"]),
        kind=MediaKind(row["kind"]),
        mime_type=row["mime_type"],
        duration_ms=row["duration_ms"],
        source=row["source"],
    )


def _segment(row: sqlite3.Row) -> SegmentSignal:
    return SegmentSignal(
        timestep=row["timestep"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        attention=row["attention"],
        engagement=row["engagement"],
        arousal=row["arousal"],
        confidence=row["confidence"],
        mesh=row["mesh"],
        vertex_count=row["vertex_count"],
        npz_path=Path(row["npz_path"]),
        frame_path=Path(row["frame_path"]) if row["frame_path"] else None,
    )
