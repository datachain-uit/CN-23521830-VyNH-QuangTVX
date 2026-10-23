"""Headroom QĐ12 chỉ dùng FullTrain-B/C02 và đúng seed đã định."""

import pytest

from skq_exp.experiments.headroom import choose_headroom_budget


def _values():
    return {
        0.005: {11: 0.76, 29: 0.77, 47: 0.765},
        0.01: {11: 0.785, 29: 0.79, 47: 0.787},
        0.02: {11: 0.795, 29: 0.796, 47: 0.794},
        0.05: {11: 0.799, 29: 0.80, 47: 0.798},
    }


def test_headroom_selects_largest_eligible_ratio() -> None:
    result = choose_headroom_budget(full_train_b_ul=0.80, c02_dev_ul=_values())
    assert result["rho_star"] == 0.01
    assert result["evidence_split"] == "dev"
    assert result["allowed_inputs"] == ["FullTrain-B", "C02"]
    assert result["secondary_descriptive_ratio"] == 0.05


def test_headroom_falls_back_to_smallest_when_none_eligible() -> None:
    values = {
        ratio: {seed: 0.799 for seed in (11, 29, 47)}
        for ratio in (0.005, 0.01, 0.02, 0.05)
    }
    result = choose_headroom_budget(full_train_b_ul=0.80, c02_dev_ul=values)
    assert result["rho_star"] == 0.005
    assert result["fallback_to_smallest"] is True


def test_headroom_rejects_missing_seed_or_ratio() -> None:
    values = _values()
    values[0.01].pop(47)
    with pytest.raises(ValueError, match="đúng seed"):
        choose_headroom_budget(full_train_b_ul=0.80, c02_dev_ul=values)
    values = _values()
    values.pop(0.02)
    with pytest.raises(ValueError, match="đúng rho"):
        choose_headroom_budget(full_train_b_ul=0.80, c02_dev_ul=values)
