from __future__ import annotations

import os
from pathlib import Path


def _bootstrap_testing_environment() -> None:
    if os.environ.get("BROKI_TESTING") != "1":
        os.environ["BROKI_TESTING"] = "1"
    os.environ.setdefault("BROKI_LIVE_VALIDATE", "0")
    os.environ.setdefault("BROKI_ENABLE_WORKER", "0")


_bootstrap_testing_environment()
