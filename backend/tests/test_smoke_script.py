from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = BACKEND_DIR / "scripts" / "smoke.sh"


def test_smoke_script_exists_and_is_executable() -> None:
    assert SMOKE_SCRIPT.exists(), f"Missing smoke script: {SMOKE_SCRIPT}"
    assert os.access(SMOKE_SCRIPT, os.X_OK), "smoke.sh is not executable"


def test_smoke_script_runs_without_live_or_e2e_tests() -> None:
    completed = subprocess.run(
        [str(SMOKE_SCRIPT)],
        cwd=str(BACKEND_DIR),
        env={**os.environ, "PYTHONPATH": "backend"},
    )
    assert completed.returncode == 0, f"smoke.sh failed with exit code {completed.returncode}"
