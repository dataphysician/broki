from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import httpx

from brainrot_guard.models import VLMDecomposition
from brainrot_guard.vlm import VLMDecompositionRequest
from brainrot_guard.vlm.risk_policy import normalize_decomposition_fields


class VLMAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class VLMProviderConfig:
    provider: str
    model: str
    endpoint: str
    api_key: str


class HTTPVLMDecomposer:
    def __init__(
        self,
        config: VLMProviderConfig,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=timeout)

    def decompose(self, request: VLMDecompositionRequest) -> VLMDecomposition:
        payload = build_provider_payload(request)
        response = self.client.post(
            self.config.endpoint,
            headers=self._headers(),
            json=self._request_body(payload),
        )
        if response.status_code >= 400:
            raise VLMAdapterError(
                f"{self.config.provider} VLM request failed with {response.status_code}: {response.text}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise VLMAdapterError("VLM provider returned non-JSON response") from exc
        return parse_decomposition_payload(data)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.provider != "gemini":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _request_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.provider == "gemini":
            return {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": json.dumps(payload, sort_keys=True)}],
                    }
                ],
                "generationConfig": {"response_mime_type": "application/json"},
            }
        return {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(payload, sort_keys=True),
                }
            ],
            "response_format": {"type": "json_object"},
        }


def build_provider_payload(request: VLMDecompositionRequest) -> dict[str, Any]:
    return {
        "instruction": (
            "Return structured JSON fields for theme, pacing, scene changes, contrast, "
            "sound effects, educational value, emotional hooks, novelty, repetition, "
            "and brainrot risk. Treat TRIBE output as a model-derived proxy, not measured brain activity."
        ),
        "media": _file_evidence(request.media.path, role="local_media", mime_type=request.media.mime_type),
        "segments": [
            {
                "timestep": segment.timestep,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "attention": segment.attention,
                "engagement": segment.engagement,
                "arousal": segment.arousal,
                "confidence": segment.confidence,
                "mesh": segment.mesh,
                "npz_artifact": segment.npz_path.name,
                "frame_artifact": segment.frame_path.name if segment.frame_path else None,
            }
            for segment in request.segments
        ],
        "plotbrain_frames": [
            _file_evidence(path, role="tribe_plotbrain_frame", mime_type="image/png")
            for path in request.frame_paths
        ],
    }


def parse_decomposition_payload(payload: Any) -> VLMDecomposition:
    if isinstance(payload, dict) and _looks_like_decomposition(payload):
        return VLMDecomposition.model_validate(normalize_decomposition_fields(payload))
    if isinstance(payload, dict) and isinstance(payload.get("decomposition"), dict):
        return VLMDecomposition.model_validate(normalize_decomposition_fields(payload["decomposition"]))
    text = _provider_text(payload)
    if text is not None:
        return VLMDecomposition.model_validate(normalize_decomposition_fields(_parse_json_text(text)))
    raise VLMAdapterError("VLM provider response did not contain a structured decomposition")


def _file_evidence(path: Path, *, role: str, mime_type: str) -> dict[str, Any]:
    if not path.exists():
        if role == "local_media":
            raise VLMAdapterError(f"local media file is missing: {path.name}")
        raise VLMAdapterError(f"PlotBrain frame artifact is missing: {path.name}")
    data = path.read_bytes()
    return {
        "role": role,
        "filename": path.name,
        "mime_type": mime_type,
        "byte_count": len(data),
        "sha256": sha256(data).hexdigest(),
        "base64": b64encode(data).decode("ascii"),
    }


def _looks_like_decomposition(value: dict[str, Any]) -> bool:
    required = {
        "theme",
        "pacing_score",
        "scene_change_cadence_hz",
        "contrast_score",
        "sound_effect_density",
        "educational_value",
        "emotional_hook_score",
        "novelty_score",
        "repetition_score",
    }
    return required.issubset(value)


def _provider_text(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise VLMAdapterError("VLM provider text did not contain valid JSON") from exc
    if not isinstance(value, dict):
        raise VLMAdapterError("VLM provider JSON must be an object")
    return value


__all__ = [
    "HTTPVLMDecomposer",
    "VLMAdapterError",
    "VLMProviderConfig",
    "build_provider_payload",
    "parse_decomposition_payload",
]
