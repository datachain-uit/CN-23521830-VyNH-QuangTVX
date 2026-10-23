"""Sáu contract test đúng sản phẩm Phụ lục B Review 22/10/2026."""

import numpy as np
import pytest

from skq_exp.data.cq_patch_1022 import (
    concentration,
    course_families,
    family_split,
    mass_weights,
    sqrt_allocation,
    temporal_overlap,
)


def test_families_merge_reofferings_and_keep_singletons() -> None:
    sets = {
        "C1": set(range(100)),
        "C2": set(range(95)) | {200},
        "C3": set(range(50, 150)),
        "C4": {1, 2, 3},
        "C5": set(range(1000, 1010)),
    }
    families = course_families(sets)
    assert families["C1"] == families["C2"]
    assert families["C3"] != families["C1"]
    assert families["C4"] != families["C1"]
    assert families["C5"] != families["C1"]


def test_families_are_transitive() -> None:
    sets = {
        "A": set(range(100)),
        "B": set(range(1, 101)),
        "C": set(range(2, 102)),
    }
    assert len(set(course_families(sets, jaccard=0.95).values())) == 1


def _families(n: int = 200) -> list[dict]:
    rng = np.random.default_rng(0)
    starts = rng.integers(0, 480, n)
    return [{
        "course_family_id": f"F{index:03d}",
        "family_start": f"2025-{1 + start // 31:02d}-{1 + start % 28:02d}",
        "family_end": f"2026-{1 + start // 31:02d}-{1 + start % 28:02d}",
        "n_enroll_labeled": int(rng.lognormal(3, 1.5)) + 1,
        "has_E": bool(rng.random() < 0.3),
    } for index, start in enumerate(starts)]


def test_family_split_temporal_test_and_disjoint() -> None:
    rows = _families()
    split = family_split(rows)
    assert set(split) == {row["course_family_id"] for row in rows}
    assert list(split.values()).count("test") == 40
    assert list(split.values()).count("dev") == 20
    assert family_split(rows) == split
    overlap = temporal_overlap(rows, split)
    assert 0.0 <= overlap["overlap_ratio"] <= 1.0


def test_concentration() -> None:
    summary = concentration(["large"] * 16_557 + [f"c{i}" for i in range(14_471)])
    assert summary["largest_share"] > 0.5
    assert summary["effective_families"] < 4


def test_sqrt_allocation_floor_cap_and_total() -> None:
    capacity = np.asarray([802, 31_566, 96_039] * 4)
    quota = sqrt_allocation(capacity, 2_568)
    assert quota.sum() == 2_568
    assert np.all(quota <= capacity)
    assert np.all(quota >= 20)
    assert quota[0] >= 30
    small = sqrt_allocation(np.asarray([5, 1000, 1000]), 300)
    assert small[0] == 5 and small.sum() == 300


def test_mass_weights_preserve_prior_and_infeasible_floor_fails() -> None:
    capacity = np.asarray([802, 31_566, 96_039])
    quota = sqrt_allocation(capacity, 642)
    weights = mass_weights(capacity, quota)
    assert np.allclose(weights * quota, capacity)
    with pytest.raises(ValueError, match="floor"):
        sqrt_allocation(capacity, 40, floor=20)
