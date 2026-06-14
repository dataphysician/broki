#!/usr/bin/env python3
"""Standalone CLI to seed a SQLite database with balanced caregiver calibration rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add backend to sys.path so brainrot_guard is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainrot_guard.fixtures.caregiver_seed import seed_caregiver_calibration
from brainrot_guard.repository import Repository


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed caregiver calibration rows into a SQLite database")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--min-per-label", type=int, default=3, help="Minimum approve/disapprove rows (default: 3)")
    args = parser.parse_args()
    db_path = Path(args.db_path)
    repo = Repository(db_path)
    repo.initialize()
    result = seed_caregiver_calibration(repo, min_per_label=args.min_per_label)
    print(json.dumps({"db_path": str(db_path), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
