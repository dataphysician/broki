import json

import brainrot_guard.__main__ as cli
from brainrot_guard.learning_validation import validate_learning_calibration


def test_learning_calibration_validation_proves_caregiver_labels_move_priors() -> None:
    report = validate_learning_calibration(random_seed=11)

    assert report["ready"] is True
    assert report["feedback_labels"] == ["disapprove", "approve"]
    assert report["similar_disapproved_profile"]["thresholds"]["engagement"] < 0.8
    assert report["similar_disapproved_profile"]["thresholds"]["risk"] < 0.7
    assert report["similar_disapproved_profile"]["skip_recommendation"]["should_skip"] is True
    assert report["educational_profile"]["skip_recommendation"]["should_skip"] is False
    assert report["telemetry_fields_present"] == []
    assert report["message"] == "ready"


def test_learning_calibration_validation_is_seeded_and_repeatable() -> None:
    first = validate_learning_calibration(random_seed=7)
    second = validate_learning_calibration(random_seed=7)

    assert first["similar_disapproved_profile"]["thresholds"]["thompson_sample"] == second[
        "similar_disapproved_profile"
    ]["thresholds"]["thompson_sample"]


def test_cli_validate_learning_prints_json(capsys) -> None:
    assert cli.main(["validate-learning", "--random-seed", "5"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["feedback_example_count"] == 2
