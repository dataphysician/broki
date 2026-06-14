from __future__ import annotations

import json
from pathlib import Path

from brainrot_guard.learning_validation import validate_caregiver_calibration_dataset
from brainrot_guard.repository import Repository
from brainrot_guard.fixtures.caregiver_seed import seed_caregiver_calibration


def test_seeder_inserts_balanced_approve_and_disapprove_records(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "calib.sqlite3")
    repo.initialize()
    result = seed_caregiver_calibration(repo)

    assert result["inserted"] == 8
    assert result["approve"] == 4
    assert result["disapprove"] == 4
    assert repo.count_feedback_examples() == 8


def test_seeded_database_passes_validate_caregiver_calibration_dataset(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "calib.sqlite3")
    repo.initialize()
    seed_caregiver_calibration(repo)

    report = validate_caregiver_calibration_dataset(repo)
    assert report["ready"] is True


def test_seeder_rejects_min_per_label_below_two(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "calib.sqlite3")
    repo.initialize()

    try:
        seed_caregiver_calibration(repo, min_per_label=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for min_per_label=1")


def test_seeder_overwrites_existing_records_idempotently(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "calib.sqlite3")
    repo.initialize()
    seed_caregiver_calibration(repo)
    seed_caregiver_calibration(repo)

    assert repo.count_feedback_examples() == 8
