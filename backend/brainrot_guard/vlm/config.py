from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import quote

from brainrot_guard.vlm.adapters import VLMProviderConfig


def provider_config_from_env(environ: Mapping[str, str]) -> VLMProviderConfig:
    provider = environ.get("VLM_PROVIDER", "minicpm").strip().lower()
    if provider == "gemini":
        api_key = _required(environ, "GEMINI_API_KEY")
        model = environ.get("GEMINI_MODEL", "gemini-flash")
        endpoint = environ.get(
            "GEMINI_API_BASE_URL",
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model)}:generateContent?key={quote(api_key)}",
        )
        return VLMProviderConfig(provider=provider, model=model, endpoint=endpoint, api_key=api_key)
    if provider == "xai":
        api_key = _required(environ, "XAI_API_KEY")
        model = environ.get("XAI_MODEL", "grok-vision")
        endpoint = environ.get("XAI_API_BASE_URL", "https://api.x.ai/v1/chat/completions")
        return VLMProviderConfig(provider=provider, model=model, endpoint=endpoint, api_key=api_key)
    if provider == "minicpm":
        api_key = _required(environ, "MINICPM_API_KEY")
        model = environ.get("MINICPM_MODEL", "minicpm-v")
        endpoint = _required(environ, "MINICPM_API_BASE_URL")
        return VLMProviderConfig(provider=provider, model=model, endpoint=endpoint, api_key=api_key)
    raise RuntimeError("Unsupported VLM_PROVIDER. Supported providers: minicpm, gemini, xai.")


def _required(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key)
    if not value:
        raise RuntimeError(f"{key} is required when VLM is enabled")
    return value


__all__ = ["provider_config_from_env"]
