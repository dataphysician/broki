from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp

import numpy as np

from brainrot_guard.models import SegmentSignal, VLMDecomposition


@dataclass(frozen=True)
class FeedbackExample:
    media_id: str
    label: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class SkipRecommendation:
    probability: float
    uncertainty: float
    should_skip: bool
    reason: str

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "probability": self.probability,
            "uncertainty": self.uncertainty,
            "should_skip": self.should_skip,
            "reason": self.reason,
        }


def feature_vector(segments: list[SegmentSignal], vlm: VLMDecomposition) -> tuple[float, ...]:
    max_engagement = max((s.engagement for s in segments), default=0.0)
    max_arousal = max((s.arousal for s in segments), default=0.0)
    mean_attention = sum((s.attention for s in segments), 0.0) / max(1, len(segments))
    return (
        max_engagement,
        max_arousal,
        mean_attention,
        vlm.risk_score,
        vlm.pacing_score,
        vlm.contrast_score,
        vlm.sound_effect_density,
        1.0 - vlm.educational_value,
        vlm.emotional_hook_score,
        vlm.novelty_score,
        vlm.repetition_score,
    )


def recommend_skip(
    target: tuple[float, ...],
    examples: list[FeedbackExample],
    *,
    threshold: float = 0.8,
) -> SkipRecommendation:
    if not examples:
        return SkipRecommendation(probability=0.5, uncertainty=0.5, should_skip=False, reason="no_parent_feedback")
    weighted_disapprove = 0.25
    weighted_total = 0.5
    for example in examples:
        distance = _squared_distance(target, example.features)
        weight = exp(-4.0 * distance)
        weighted_total += weight
        if example.label == "disapprove":
            weighted_disapprove += weight
    probability = weighted_disapprove / weighted_total
    uncertainty = 1.0 / weighted_total
    return SkipRecommendation(
        probability=probability,
        uncertainty=uncertainty,
        should_skip=probability >= threshold,
        reason="similar_to_disapproved_content" if probability >= threshold else "below_skip_threshold",
    )


def learned_thresholds(
    target: tuple[float, ...],
    examples: list[FeedbackExample],
    *,
    base_engagement: float = 0.8,
    base_risk: float = 0.7,
    random_seed: int | None = None,
) -> dict[str, float | str]:
    if not examples:
        return {
            "engagement": base_engagement,
            "risk": base_risk,
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
            "beta_disapproval_mean": 0.5,
            "gp_disapproval_mean": 0.5,
            "gp_disapproval_variance": 0.25,
            "thompson_sample": 0.5,
            "source": "default_prior",
        }
    disapprove_count = sum(1 for example in examples if example.label == "disapprove")
    approve_count = sum(1 for example in examples if example.label == "approve")
    beta_alpha = 1.0 + disapprove_count
    beta_beta = 1.0 + approve_count
    beta_mean = beta_alpha / (beta_alpha + beta_beta)
    seed = random_seed if random_seed is not None else _stable_threshold_seed(target, examples)
    thompson_sample = _thompson_sample(beta_alpha, beta_beta, random_seed=seed)
    gp = _gp_disapproval_posterior(target, examples)
    signal = (thompson_sample + gp.mean) / 2.0
    if signal >= 0.55:
        engagement = base_engagement - (0.08 * signal)
        risk = base_risk - (0.07 * signal)
    else:
        engagement = base_engagement + (0.04 * (0.55 - signal))
        risk = base_risk + (0.04 * (0.55 - signal))
    return {
        "engagement": max(0.55, min(0.95, engagement)),
        "risk": max(0.5, min(0.95, risk)),
        "beta_alpha": beta_alpha,
        "beta_beta": beta_beta,
        "beta_disapproval_mean": beta_mean,
        "gp_disapproval_mean": gp.mean,
        "gp_disapproval_variance": gp.variance,
        "thompson_sample": thompson_sample,
        "source": "caregiver_feedback",
    }


@dataclass(frozen=True)
class _Posterior:
    mean: float
    variance: float


def _thompson_sample(alpha: float, beta: float, *, random_seed: int | None) -> float:
    rng = np.random.default_rng(random_seed)
    return float(rng.beta(alpha, beta))


def _stable_threshold_seed(target: tuple[float, ...], examples: list[FeedbackExample]) -> int:
    payload = {
        "target": [round(value, 6) for value in target],
        "examples": [
            {
                "label": example.label,
                "features": [round(value, 6) for value in example.features],
            }
            for example in examples
        ],
    }
    digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _gp_disapproval_posterior(target: tuple[float, ...], examples: list[FeedbackExample]) -> _Posterior:
    if not target:
        return _Posterior(mean=0.5, variance=0.25)
    x_train = np.array([example.features for example in examples], dtype=np.float64)
    y_train = np.array([1.0 if example.label == "disapprove" else 0.0 for example in examples], dtype=np.float64)
    x_target = np.array(target, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        return _Posterior(mean=0.5, variance=0.25)
    k_train = _rbf_kernel(x_train, x_train) + np.eye(len(examples)) * 0.08
    k_target = _rbf_kernel(x_train, x_target.reshape(1, -1)).reshape(-1)
    centered = y_train - 0.5
    try:
        weights = np.linalg.solve(k_train, centered)
        solved = np.linalg.solve(k_train, k_target)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(k_train) @ centered
        solved = np.linalg.pinv(k_train) @ k_target
    mean = 0.5 + float(k_target @ weights)
    variance = float(max(0.0, 1.0 - k_target @ solved))
    return _Posterior(mean=max(0.0, min(1.0, mean)), variance=min(1.0, variance))


def _rbf_kernel(left: np.ndarray, right: np.ndarray, *, length_scale: float = 0.75) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    return np.exp(-distances / (2.0 * length_scale * length_scale))


def _squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=False))
