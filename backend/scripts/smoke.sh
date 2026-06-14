#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=backend pytest -m "not live and not e2e_browser" --tb=short --ignore=tests/test_smoke_script.py --deselect=tests/test_rebuilt_baseline.py::test_static_review_console_advances_image_and_text_timeline
exit $?
