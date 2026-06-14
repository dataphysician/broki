from __future__ import annotations

from typing import Any


def normalize_decomposition_fields(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "risk_score" not in normalized:
        normalized["risk_score"] = _derived_risk_score(normalized)
    if "risk_rationale" not in normalized:
        normalized["risk_rationale"] = _derived_rationale(normalized, float(normalized["risk_score"]))
    return normalized


def _derived_risk_score(payload: dict[str, Any]) -> float:
    educational_value = _score(payload, "educational_value")
    stimulation = (
        _score(payload, "pacing_score")
        + _scene_change_score(payload)
        + _score(payload, "contrast_score")
        + _score(payload, "sound_effect_density")
    ) / 4.0
    reward_pull = (
        _score(payload, "emotional_hook_score")
        + _score(payload, "novelty_score")
        + _score(payload, "repetition_score")
    ) / 3.0
    low_value = 1.0 - educational_value
    risk = (0.42 * stimulation) + (0.38 * reward_pull) + (0.2 * low_value)
    if educational_value >= 0.8 and stimulation <= 0.35:
        risk *= 0.45
    if educational_value <= 0.2 and stimulation >= 0.7 and reward_pull >= 0.7:
        risk = max(risk, 0.82)
    return round(max(0.0, min(1.0, risk)), 4)


def _derived_rationale(payload: dict[str, Any], risk_score: float) -> str:
    educational_value = _score(payload, "educational_value")
    stimulation = (
        _score(payload, "pacing_score")
        + _scene_change_score(payload)
        + _score(payload, "contrast_score")
        + _score(payload, "sound_effect_density")
    ) / 4.0
    reward_pull = (
        _score(payload, "emotional_hook_score")
        + _score(payload, "novelty_score")
        + _score(payload, "repetition_score")
    ) / 3.0
    reasons = []
    if stimulation >= 0.65:
        reasons.append("high stimulation")
    if reward_pull >= 0.65:
        reasons.append("high reward-pull pairing")
    if educational_value <= 0.25:
        reasons.append("low educational value")
    if educational_value >= 0.75:
        reasons.append("strong educational value")
    if not reasons:
        reasons.append("mixed structured content signals")
    return f"field-pair policy risk={risk_score:.2f}: " + ", ".join(reasons)


def _scene_change_score(payload: dict[str, Any]) -> float:
    cadence = max(0.0, _number(payload.get("scene_change_cadence_hz")))
    return max(0.0, min(1.0, cadence / 0.4))


def _score(payload: dict[str, Any], key: str) -> float:
    return max(0.0, min(1.0, _number(payload.get(key))))


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["normalize_decomposition_fields"]
