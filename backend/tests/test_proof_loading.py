from __future__ import annotations

import json

import pytest

from brainrot_guard.proofs import EVIDENCE_DIR_DEFAULT, load_all_proofs, load_proof, save_proof


def test_load_proof_returns_none_when_file_absent(tmp_path: Path) -> None:
    assert load_proof(tmp_path, "analysis") is None


def test_save_and_load_proof_round_trip(tmp_path: Path) -> None:
    payload = {"foo": 1, "bar": [1, 2]}
    save_proof(tmp_path, "browser", payload)
    loaded = load_proof(tmp_path, "browser")
    assert loaded == payload


def test_load_all_proofs_returns_empty_dict_for_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    loaded = load_all_proofs(missing)
    assert loaded == {"analysis": None, "browser": None, "vlm_probe": None}


def test_load_all_proofs_returns_each_proof_when_files_present(tmp_path: Path) -> None:
    save_proof(tmp_path, "analysis", {"analysis": True})
    save_proof(tmp_path, "browser", {"browser": True})
    loaded = load_all_proofs(tmp_path)
    assert loaded["analysis"] == {"analysis": True}
    assert loaded["browser"] == {"browser": True}
    assert loaded["vlm_probe"] is None


def test_save_proof_creates_missing_evidence_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested"
    path = save_proof(target, "analysis", {"ok": True})
    assert path.exists()
    assert path.parent == target
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
