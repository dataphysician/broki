from pathlib import Path

import httpx
import numpy as np
import pytest

from brainrot_guard.artifacts import render_demo_frame, write_segment_npz
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.models import SegmentSignal
from brainrot_guard.vlm import VLMDecompositionRequest
from brainrot_guard.vlm.adapters import (
    HTTPVLMDecomposer,
    VLMAdapterError,
    VLMProviderConfig,
    build_provider_payload,
    parse_decomposition_payload,
)


def test_parse_decomposition_payload_accepts_direct_json_fields() -> None:
    decomposition = parse_decomposition_payload(
        {
            "theme": "counting song",
            "pacing_score": 0.2,
            "scene_change_cadence_hz": 0.1,
            "contrast_score": 0.3,
            "sound_effect_density": 0.2,
            "educational_value": 0.9,
            "emotional_hook_score": 0.2,
            "novelty_score": 0.2,
            "repetition_score": 0.4,
            "risk_score": 0.1,
            "risk_rationale": "educational structure dominates",
        }
    )

    assert decomposition.theme == "counting song"
    assert decomposition.educational_value == 0.9


def test_parse_decomposition_payload_extracts_json_from_gemini_candidates() -> None:
    decomposition = parse_decomposition_payload(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "```json\n"
                                '{"theme":"toy loop","pacing_score":0.9,'
                                '"scene_change_cadence_hz":0.8,"contrast_score":0.8,'
                                '"sound_effect_density":0.7,"educational_value":0.1,'
                                '"emotional_hook_score":0.9,"novelty_score":0.8,'
                                '"repetition_score":0.9,"risk_score":0.92,'
                                '"risk_rationale":"high reward density"}\n```'
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert decomposition.theme == "toy loop"
    assert decomposition.risk_score == 0.92


def test_parse_decomposition_payload_derives_high_risk_from_brainrot_field_pairings() -> None:
    decomposition = parse_decomposition_payload(
        {
            "theme": "slice of life toy loop",
            "pacing_score": 0.92,
            "scene_change_cadence_hz": 0.4,
            "contrast_score": 0.88,
            "sound_effect_density": 0.82,
            "educational_value": 0.05,
            "emotional_hook_score": 0.9,
            "novelty_score": 0.86,
            "repetition_score": 0.91,
        }
    )

    assert decomposition.risk_score >= 0.8
    assert "high stimulation" in decomposition.risk_rationale
    assert "low educational value" in decomposition.risk_rationale


def test_parse_decomposition_payload_derives_low_risk_for_educational_field_pairings() -> None:
    decomposition = parse_decomposition_payload(
        {
            "theme": "counting operations song",
            "pacing_score": 0.22,
            "scene_change_cadence_hz": 0.05,
            "contrast_score": 0.28,
            "sound_effect_density": 0.12,
            "educational_value": 0.93,
            "emotional_hook_score": 0.2,
            "novelty_score": 0.2,
            "repetition_score": 0.35,
        }
    )

    assert decomposition.risk_score <= 0.25
    assert "educational value" in decomposition.risk_rationale


def test_http_vlm_decomposer_posts_provider_payload_and_parses_response(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer secret"
        body = request.read().decode("utf-8")
        assert "TRIBE output" in body
        assert "story.txt" in body
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"theme":"fast clips","pacing_score":0.9,'
                            '"scene_change_cadence_hz":0.9,"contrast_score":0.8,'
                            '"sound_effect_density":0.8,"educational_value":0.05,'
                            '"emotional_hook_score":0.9,"novelty_score":0.8,'
                            '"repetition_score":0.95,"risk_score":0.93,'
                            '"risk_rationale":"low-value reward loop"}'
                        }
                    }
                ]
            },
        )

    decomposer = HTTPVLMDecomposer(
        VLMProviderConfig(
            provider="xai",
            model="grok-vision",
            endpoint="https://api.x.ai/v1/chat/completions",
            api_key="secret",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = decomposer.decompose(_request(tmp_path))

    assert len(calls) == 1
    assert result.theme == "fast clips"
    assert result.risk_score == 0.93


def test_http_vlm_decomposer_raises_provider_error(tmp_path: Path) -> None:
    decomposer = HTTPVLMDecomposer(
        VLMProviderConfig(
            provider="gemini",
            model="gemini-flash",
            endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash:generateContent",
            api_key="secret",
        ),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429, text="quota"))),
    )

    with pytest.raises(VLMAdapterError, match="429"):
        decomposer.decompose(_request(tmp_path))


def test_build_provider_payload_still_avoids_local_paths(tmp_path: Path) -> None:
    request = _request(tmp_path)

    payload = build_provider_payload(request)

    assert str(tmp_path) not in str(payload)


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
