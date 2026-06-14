from __future__ import annotations

from typing import Any

from brainrot_guard.learning_validation import _brainrot_like_profile, _educational_profile
from brainrot_guard.repository import Repository

# 3 full-decomposition approve profiles (11-element tuples), drawn from _educational_profile
# 3 full-decomposition disapprove profiles (11-element tuples), drawn from _brainrot_like_profile
# 1 segment-response approve profile (3-element tuple)
# 1 segment-response disapprove profile (3-element tuple)

_FULL_DECOMPOSITION_SEED: list[tuple[str, tuple[float, ...]]] = [
    ("approve-1", _educational_profile()),
    ("approve-2", _educational_profile()),
    ("approve-3", _educational_profile()),
    ("disapprove-1", _brainrot_like_profile()),
    ("disapprove-2", _brainrot_like_profile()),
    ("disapprove-3", _brainrot_like_profile()),
]

_SEGMENT_RESPONSE_SEED: list[tuple[str, tuple[float, ...]]] = [
    ("segment-approve-1", (0.5, 0.5, 0.5)),
    ("segment-disapprove-1", (0.9, 0.9, 0.9)),
]


def seed_caregiver_calibration(
    repository: Repository, *, min_per_label: int = 3
) -> dict[str, int]:
    """Insert balanced caregiver calibration rows.

    Returns a dict with keys: "inserted", "approve", "disapprove".
    Raises ValueError if min_per_label < 2.
    """
    if min_per_label < 2:
        raise ValueError("min_per_label must be at least 2 to satisfy validate_caregiver_calibration_dataset thresholds")
    inserted = 0
    approve_count = 0
    disapprove_count = 0
    for media_id, features in _FULL_DECOMPOSITION_SEED:
        label = media_id.split("-")[0]  # "approve" or "disapprove"
        repository.record_feedback_example(media_id, label, features)
        inserted += 1
        if label == "approve":
            approve_count += 1
        else:
            disapprove_count += 1
    for media_id, features in _SEGMENT_RESPONSE_SEED:
        # media_id format: "segment-{label}-1" -> label is the middle part
        label = media_id.split("-")[1]  # "approve" or "disapprove"
        repository.record_feedback_example(media_id, label, features)
        inserted += 1
        if label == "approve":
            approve_count += 1
        else:
            disapprove_count += 1
    if approve_count < min_per_label or disapprove_count < min_per_label:
        raise RuntimeError(
            f"seeder produced {approve_count} approve and {disapprove_count} disapprove, "
            f"but min_per_label={min_per_label} requires at least that many of each"
        )
    return {"inserted": inserted, "approve": approve_count, "disapprove": disapprove_count}


__all__ = ["seed_caregiver_calibration"]
