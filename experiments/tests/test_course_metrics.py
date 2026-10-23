"""Endpoint CQ giữ đúng cấp enrollment/course và PCC là chính."""

from __future__ import annotations

import pandas as pd
import pytest

from skq_exp.training.course_metrics import (
    class_conditional_means,
    class_prevalence,
    cq_endpoints,
    prior_correct_predictions,
    prior_correct_probabilities,
    validate_private_predictions,
)


def _predictions() -> pd.DataFrame:
    rows = []
    enrollment = [
        (10, 1, 0.1, "E", [0.8, 0.15, 0.05]),
        (10, 2, 0.3, "G", [0.1, 0.8, 0.1]),
        (20, 3, 0.6, "A", [0.05, 0.15, 0.8]),
        (20, 4, 0.5, "A", [0.05, 0.2, 0.75]),
    ]
    for phase in ("P1", "P2", "P3", "P4"):
        for course, learner, q, label, probability in enrollment:
            rows.append({
                "run_id": "run-1", "course_id": course,
                "course_root_id": course, "learner_id": learner, "phase": phase,
                "q_true": q, "y_true": label, "y_pred": label,
                "p_E": probability[0], "p_G": probability[1], "p_A": probability[2],
            })
    return pd.DataFrame(rows)


def test_private_prediction_schema_and_two_level_endpoints() -> None:
    predictions = _predictions()
    assert validate_private_predictions(predictions)["row_level_visibility"] == "private_only"
    mu = {"E": 0.1, "G": 0.3, "A": 0.55}
    result = cq_endpoints(predictions, mu, n_min=2)
    assert result["ul_mean_p1_p4"] == pytest.approx(1.0)
    assert result["ul_p1_secondary"] == pytest.approx(1.0)
    assert result["ec_mean_p1_p4"] is not None
    assert result["ec_missing_phase_count"] == 0


def test_n_min_can_make_ec_missing_but_not_zero() -> None:
    result = cq_endpoints(_predictions(), {"E": 0.1, "G": 0.3, "A": 0.55}, n_min=3)
    assert result["ec_mean_p1_p4"] is None
    assert result["ec_missing_phase_count"] == 4


def test_class_means_use_unique_train_enrollments_and_validate_order() -> None:
    train = pd.DataFrame({"q": [0.1, 0.3, 0.6], "y": ["E", "G", "A"]})
    assert class_conditional_means(train) == {"E": 0.1, "G": 0.3, "A": 0.6}
    bad = pd.DataFrame({"q": [0.5, 0.3, 0.1], "y": ["E", "G", "A"]})
    with pytest.raises(ValueError, match="mu_E"):
        class_conditional_means(bad)


def test_duplicate_prediction_rows_are_rejected() -> None:
    predictions = _predictions()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="trùng"):
        validate_private_predictions(duplicated)


def test_integer_prediction_labels_are_normalized_for_ul() -> None:
    predictions = _predictions()
    mapping = {"E": 0, "G": 1, "A": 2}
    predictions["y_true"] = predictions["y_true"].map(mapping)
    predictions["y_pred"] = predictions["y_pred"].map(mapping)
    result = cq_endpoints(predictions, {"E": 0.1, "G": 0.3, "A": 0.55}, n_min=2)
    assert result["ul_mean_p1_p4"] == pytest.approx(1.0)


def test_a8_prior_correction_is_identity_for_equal_priors() -> None:
    probabilities = _predictions().loc[:3, ["p_E", "p_G", "p_A"]].to_numpy()
    prior = {"E": 0.25, "G": 0.25, "A": 0.5}
    assert prior_correct_probabilities(probabilities, prior, prior) == pytest.approx(probabilities)


def test_a8_prior_correction_uses_phase_target_and_does_not_mutate_input() -> None:
    predictions = _predictions()
    original = predictions.copy(deep=True)
    source = {"E": 1 / 3, "G": 1 / 3, "A": 1 / 3}
    targets = {
        "P1": {"E": 0.8, "G": 0.1, "A": 0.1},
        "P2": source,
        "P3": source,
        "P4": source,
    }
    corrected = prior_correct_predictions(predictions, source, targets)
    assert predictions.equals(original)
    assert corrected.loc[corrected["phase"] == "P1", "p_E"].mean() > predictions.loc[predictions["phase"] == "P1", "p_E"].mean()
    assert class_prevalence(["E", "G", "A", "A"]) == pytest.approx(
        {"E": 0.25, "G": 0.25, "A": 0.5}
    )
