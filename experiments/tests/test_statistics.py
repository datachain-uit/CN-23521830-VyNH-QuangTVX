"""Kiểm dấu effect, pairing course root và Holm family."""

import numpy as np
import pytest

from skq_exp.training.statistics import (
    ECUnit,
    ULUnit,
    holm_adjust,
    paired_course_bootstrap_ec,
    paired_course_bootstrap_ul,
)


def test_ul_positive_effect_favors_a_and_is_deterministic() -> None:
    course = np.repeat(np.arange(10), 3)
    truth = np.tile(np.array(["E", "G", "A"]), 10)
    pred_a = truth.copy()
    pred_b = np.repeat("G", len(truth))
    unit = ULUnit(truth, pred_a, pred_b, course, "seed11-P1")
    first = paired_course_bootstrap_ul((unit,), labels=("E", "G", "A"), n_boot=200, seed=7)
    second = paired_course_bootstrap_ul((unit,), labels=("E", "G", "A"), n_boot=200, seed=7)
    assert first == second
    assert first["point_estimate"] > 0
    assert first["effect_name"] == "delta_ul_a_minus_b"


def test_v4_bootstrap_records_course_family_unit() -> None:
    unit = ULUnit(
        y_true=np.asarray([0, 1, 0, 1]),
        pred_a=np.asarray([0, 1, 0, 1]),
        pred_b=np.asarray([0, 0, 0, 1]),
        course_root_id=np.asarray(["f1", "f1", "f2", "f2"]),
        unit_id="v4",
    )
    result = paired_course_bootstrap_ul(
        (unit,), labels=(0, 1), n_boot=20, seed=7,
        resampling_unit="course_family_id",
    )
    assert result["resampling_unit"] == "course_family_id"


def test_ec_positive_effect_means_a_has_lower_error() -> None:
    unit = ECUnit(
        error_a=np.full(12, 0.05),
        error_b=np.full(12, 0.20),
        course_root_id=np.arange(12),
        unit_id="seed11-P1",
    )
    result = paired_course_bootstrap_ec((unit,), n_boot=200, seed=9)
    assert result["point_estimate"] == pytest.approx(0.15)
    assert result["effect_name"] == "delta_ec_b_minus_a"


def test_course_set_mismatch_is_not_silently_dropped() -> None:
    first = ECUnit(np.ones(3), np.ones(3), np.array([1, 2, 3]), "one")
    second = ECUnit(np.ones(3), np.ones(3), np.array([1, 2, 4]), "two")
    with pytest.raises(ValueError, match="không tự drop"):
        paired_course_bootstrap_ec((first, second), n_boot=20)


def test_holm_is_step_down_and_kept_inside_one_family() -> None:
    result = holm_adjust({"K1": 0.001, "K2": 0.02, "K3": 0.2}, alpha=0.05)
    assert result["K1"]["reject"] is True
    assert result["K2"]["reject"] is True
    assert result["K3"]["reject"] is False
    assert result["K1"]["holm_adjusted_p"] <= result["K2"]["holm_adjusted_p"]
