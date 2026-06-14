from __future__ import annotations

from pathlib import Path
from typing import Any

from brainrot_guard.ingestion.scanner import _looks_like_url_file
from brainrot_guard.repository import Repository


FORBIDDEN_TELEMETRY_TERMS = (
    "retention",
    "scroll",
    "gaze",
    "watch_time",
    "watchtime",
    "biometric",
    "pause",
    "click",
    "view_duration",
    "child_behavior",
)


def validate_privacy_boundaries(
    *,
    repository: Repository,
    media_dir: Path | None = None,
) -> dict[str, Any]:
    repository.initialize()
    forbidden_sources = _forbidden_sources(media_dir) if media_dir is not None else []
    forbidden_schema_fields = _forbidden_schema_fields(repository)
    non_binary_rejected = _non_binary_feedback_rejected(repository)
    failures = []
    if forbidden_sources:
        failures.append("YouTube/browser source placeholders are out of scope")
    if forbidden_schema_fields:
        failures.append("child behavior telemetry fields are not allowed")
    if not non_binary_rejected:
        failures.append("feedback must reject labels outside caregiver binary approve/disapprove")
    ready = not failures
    return {
        "ready": ready,
        "media_dir": str(media_dir.expanduser().resolve()) if media_dir else None,
        "forbidden_source_count": len(forbidden_sources),
        "forbidden_sources": forbidden_sources,
        "forbidden_schema_fields": forbidden_schema_fields,
        "feedback_labels_allowed": ["approve", "disapprove"],
        "non_binary_feedback_rejected": non_binary_rejected,
        "message": "ready" if ready else "; ".join(failures),
    }


def _forbidden_sources(media_dir: Path) -> list[str]:
    root = media_dir.expanduser().resolve()
    if not root.is_dir():
        return []
    forbidden = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() == ".url" or _looks_like_url_file(path):
            forbidden.append(path.name)
    return forbidden


def _forbidden_schema_fields(repository: Repository) -> list[str]:
    found = []
    with repository.connect() as conn:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in sorted(tables):
            for column in conn.execute(f"PRAGMA table_info({table})"):
                field = str(column["name"]).lower()
                if any(term in field for term in FORBIDDEN_TELEMETRY_TERMS):
                    found.append(f"{table}.{column['name']}")
    return sorted(found)


def _non_binary_feedback_rejected(repository: Repository) -> bool:
    rejected_feedback = _raises_value_error(lambda: repository.record_feedback("__privacy_probe__", "watched_long"))
    rejected_example = _raises_value_error(
        lambda: repository.record_feedback_example("__privacy_probe__", "watched_long", ())
    )
    return rejected_feedback and rejected_example


def _raises_value_error(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


__all__ = ["validate_privacy_boundaries"]
