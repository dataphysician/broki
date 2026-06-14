from __future__ import annotations

import json
from collections import Counter
from typing import Any

from brainrot_guard.learning import FeedbackExample, learned_thresholds, recommend_skip


SEGMENT_RESPONSE_FEATURE_COUNT = 3
FULL_DECOMPOSITION_FEATURE_COUNT = 11
VALID_FEEDBACK_LABELS = {"approve", "disapprove"}


def validate_learning_calibration(*, random_seed: int = 7) -> dict[str, Any]:
    disapproved = _brainrot_like_profile()
    approved = _educational_profile()
    examples = [
        FeedbackExample(media_id="fixture-disapproved", label="disapprove", features=disapproved),
        FeedbackExample(media_id="fixture-approved", label="approve", features=approved),
    ]
    similar = _brainrot_like_profile(engagement=0.9, risk=0.88)
    educational = _educational_profile()
    similar_thresholds = learned_thresholds(similar, examples, random_seed=random_seed)
    educational_thresholds = learned_thresholds(educational, examples, random_seed=random_seed)
    similar_skip = recommend_skip(similar, examples).as_dict()
    educational_skip = recommend_skip(educational, examples).as_dict()
    telemetry_fields = _telemetry_fields(similar_thresholds) + _telemetry_fields(educational_thresholds)
    ready = (
        similar_thresholds["engagement"] < 0.8
        and similar_thresholds["risk"] < 0.7
        and bool(similar_skip["should_skip"])
        and not bool(educational_skip["should_skip"])
        and not telemetry_fields
    )
    return {
        "ready": ready,
        "feedback_labels": [example.label for example in examples],
        "feedback_example_count": len(examples),
        "similar_disapproved_profile": {
            "thresholds": similar_thresholds,
            "skip_recommendation": similar_skip,
        },
        "educational_profile": {
            "thresholds": educational_thresholds,
            "skip_recommendation": educational_skip,
        },
        "telemetry_fields_present": telemetry_fields,
        "message": "ready" if ready else "learning calibration did not meet expected guardrails",
    }


def validate_caregiver_calibration_dataset(
    repository,
    *,
    min_examples: int = 6,
    min_approve: int = 2,
    min_disapprove: int = 2,
    min_segment_response: int = 1,
    min_full_decomposition: int = 2,
) -> dict[str, Any]:
    records = _feedback_example_records(repository)
    label_counts: Counter[str] = Counter()
    feature_scope_counts: Counter[str] = Counter()
    invalid_labels = []
    invalid_feature_scopes = []
    invalid_payloads = []
    telemetry_fields = []
    for record in records:
        label = record["label"]
        features = record["features"]
        if label in VALID_FEEDBACK_LABELS:
            label_counts[label] += 1
        else:
            invalid_labels.append(record["media_id"])
        scope = _feature_scope(features)
        if scope is None:
            invalid_feature_scopes.append(record["media_id"])
        else:
            feature_scope_counts[scope] += 1
        if record["payload_error"]:
            invalid_payloads.append(record["media_id"])
        telemetry_fields.extend(_telemetry_like_values(record))

    problems = []
    if invalid_labels or invalid_feature_scopes or invalid_payloads:
        problems.append("invalid caregiver calibration records")
    if len(records) < min_examples:
        problems.append("not enough caregiver feedback examples")
    if label_counts["approve"] < min_approve:
        problems.append("not enough approve examples")
    if label_counts["disapprove"] < min_disapprove:
        problems.append("not enough disapprove examples")
    if feature_scope_counts["segment_response"] < min_segment_response:
        problems.append("not enough segment-response examples")
    if feature_scope_counts["full_decomposition"] < min_full_decomposition:
        problems.append("not enough full-decomposition examples")
    if telemetry_fields:
        problems.append("telemetry-like caregiver records are present")

    return {
        "ready": not problems,
        "feedback_example_count": len(records),
        "minimum_examples": min_examples,
        "label_counts": dict(sorted(label_counts.items())),
        "feature_scope_counts": dict(sorted(feature_scope_counts.items())),
        "invalid_label_count": len(invalid_labels),
        "invalid_label_media_ids": invalid_labels,
        "invalid_feature_scope_count": len(invalid_feature_scopes),
        "invalid_feature_scope_media_ids": invalid_feature_scopes,
        "invalid_payload_count": len(invalid_payloads),
        "invalid_payload_media_ids": invalid_payloads,
        "telemetry_fields_present": sorted(set(telemetry_fields)),
        "message": "ready" if not problems else "; ".join(problems),
    }


def _feedback_example_records(repository) -> list[dict[str, Any]]:
    with repository.connect() as conn:
        rows = conn.execute("SELECT * FROM feedback_examples ORDER BY media_id").fetchall()
    records = []
    for row in rows:
        payload_error = None
        features: tuple[float, ...] = ()
        try:
            raw_features = json.loads(row["features_json"])
            if not isinstance(raw_features, list):
                raise ValueError("features_json must be a list")
            features = tuple(float(value) for value in raw_features)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            payload_error = str(exc)
        records.append(
            {
                "media_id": row["media_id"],
                "label": row["label"],
                "features": features,
                "payload_error": payload_error,
            }
        )
    return records


def _feature_scope(features: tuple[float, ...]) -> str | None:
    if len(features) == SEGMENT_RESPONSE_FEATURE_COUNT:
        return "segment_response"
    if len(features) == FULL_DECOMPOSITION_FEATURE_COUNT:
        return "full_decomposition"
    return None


def _telemetry_like_values(record: dict[str, Any]) -> list[str]:
    forbidden = ("retention", "scroll", "watch", "gaze", "biometric", "click", "pause")
    values = [str(record["media_id"]).lower(), str(record["label"]).lower()]
    return [value for value in values if any(token in value for token in forbidden)]


def _brainrot_like_profile(*, engagement: float = 0.92, risk: float = 0.9) -> tuple[float, ...]:
    return (
        engagement,
        0.86,
        0.72,
        risk,
        0.9,
        0.86,
        0.78,
        0.92,
        0.9,
        0.82,
        0.91,
    )


def _educational_profile() -> tuple[float, ...]:
    return (
        0.42,
        0.28,
        0.68,
        0.12,
        0.24,
        0.3,
        0.12,
        0.08,
        0.18,
        0.22,
        0.34,
    )


def _telemetry_fields(thresholds: dict[str, Any]) -> list[str]:
    forbidden = {"retention", "scroll", "gaze", "watch_time", "biometric", "pause", "click"}
    return sorted(key for key in thresholds if any(term in key for term in forbidden))


__all__ = ["validate_caregiver_calibration_dataset", "validate_learning_calibration"]
