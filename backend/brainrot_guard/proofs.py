from __future__ import annotations

import json
from pathlib import Path


EVIDENCE_DIR_DEFAULT: Path = Path(".sisyphus/evidence")
PROOF_FILENAMES: dict[str, str] = {
    "analysis": "analysis.json",
    "browser": "browser.json",
    "vlm_probe": "vlm-probe.json",
}


def load_proof(evidence_dir: Path | str, kind: str) -> dict | None:
    """Load a proof JSON from evidence_dir; return None if file is missing."""
    if kind not in PROOF_FILENAMES:
        raise ValueError(f"unknown proof kind: {kind!r}")
    path = Path(evidence_dir) / PROOF_FILENAMES[kind]
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"proof {path} must be a JSON object, got {type(value).__name__}")
    return value


def save_proof(evidence_dir: Path | str, kind: str, payload: dict) -> Path:
    """Write a proof JSON; create evidence_dir if missing; return absolute path."""
    if kind not in PROOF_FILENAMES:
        raise ValueError(f"unknown proof kind: {kind!r}")
    target = Path(evidence_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = target / PROOF_FILENAMES[kind]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return path


def load_all_proofs(evidence_dir: Path | str | None = None) -> dict[str, dict | None]:
    """Return a dict keyed by proof kind ('analysis', 'browser', 'vlm_probe')."""
    base = Path(evidence_dir) if evidence_dir is not None else EVIDENCE_DIR_DEFAULT
    return {kind: load_proof(base, kind) for kind in PROOF_FILENAMES}


__all__ = ["EVIDENCE_DIR_DEFAULT", "PROOF_FILENAMES", "load_proof", "save_proof", "load_all_proofs"]
