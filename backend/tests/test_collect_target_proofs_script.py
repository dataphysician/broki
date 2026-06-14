from __future__ import annotations

import os
import subprocess
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = BACKEND_DIR / "scripts" / "collect_target_proofs.sh"
REPO_ROOT = BACKEND_DIR.parent


def test_collect_target_proofs_sh_exists_and_is_executable() -> None:
    assert COLLECT_SCRIPT.exists(), f"Missing collect script: {COLLECT_SCRIPT}"
    assert os.access(COLLECT_SCRIPT, os.X_OK), "collect_target_proofs.sh is not executable"


def test_collect_target_proofs_sh_dry_run_prints_all_commands() -> None:
    completed = subprocess.run(
        ["bash", str(COLLECT_SCRIPT), "--media-dir", "/tmp/x", "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "backend"},
    )
    assert completed.returncode == 0, (
        f"collect_target_proofs.sh --dry-run failed with exit code {completed.returncode}\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    output = completed.stdout
    expected_commands = [
        "seed-calibration",
        "validate-hardware",
        "validate-live",
        "validate-vlm-live",
        "validate-analysis",
        "validate-browser",
        "validate-readiness",
    ]
    for command in expected_commands:
        assert command in output, f"dry-run output missing command: {command}"
