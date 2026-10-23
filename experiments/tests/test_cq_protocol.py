"""Kiểm contract CQ v1 trực tiếp từ Review 12–13/10."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skq_exp.data.cq_protocol import (
    SplitDecisionThresholds,
    choose_split_strategy,
    enrollment_table,
    phase_cutoffs,
    quality_class,
    temporal_course_split,
    validate_enrollment_table,
    validate_pool_b,
)


def test_enrollment_table_normalizes_loqv_and_preserves_within_course_labels() -> None:
    chapters = pd.DataFrame({
        "learner_id": [1, 1, 2, 2, 3],
        "course_id": [10, 10, 10, 10, 10],
        "euclid_to_111": np.array([0.1, 0.2, 0.5, 0.7, 0.3]) * np.sqrt(3),
    })
    table, audit = enrollment_table(
        chapters, unopened_chapter_policy="included_as_rows",
    )
    assert table["q"].round(6).tolist() == [0.15, 0.6, 0.3]
    assert table["y"].tolist() == ["E", "A", "G"]
    assert audit.loqv_was_divided_by_sqrt3 is True
    assert validate_enrollment_table(table)["courses_with_label_variation"] == 1


def test_quality_class_boundary_convention_is_explicit() -> None:
    assert quality_class(np.array([0.0, 0.199, 0.2, 0.399, 0.4, 1.0])).tolist() == [
        "E", "E", "G", "G", "A", "A",
    ]


def test_phase_cutoffs_follow_25_50_75_90_percent() -> None:
    cutoffs = phase_cutoffs(
        pd.Series(["2026-01-01"]), pd.Series(["2026-01-11"]),
    )
    assert cutoffs.loc[0, "P2"] == pd.Timestamp("2026-01-06")
    assert cutoffs.loc[0, "P4"] == pd.Timestamp("2026-01-10")


def test_temporal_and_sgkf_decisions_are_mutually_exclusive() -> None:
    diagnostics = {
        "timestamp_missing_rate": 0.01,
        "timestamp_order_consistent": True,
        "temporal_test_course_counts_by_class": {"E": 12, "G": 20, "A": 11},
        "dev_ci_half_width": 0.004,
    }
    thresholds = SplitDecisionThresholds(max_timestamp_missing_rate=0.02)
    assert choose_split_strategy(diagnostics, thresholds)["decision"] == "TEMPORAL_70_10_20"
    diagnostics["dev_ci_half_width"] = 0.006
    assert choose_split_strategy(diagnostics, thresholds)["decision"] == "SGKF_5_FOLD_X_2"


def test_temporal_split_keeps_all_runs_of_root_together() -> None:
    courses = pd.DataFrame({
        "course_root_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "course_start": pd.date_range("2026-01-01", periods=10, freq="D"),
    })
    split = temporal_course_split(courses)
    assert set(split.unique()) == {"train", "dev", "test"}
    assert split.index.is_unique


def test_pool_b_requires_four_snapshots_and_fixed_label() -> None:
    snapshots = pd.DataFrame([
        {"learner_id": learner, "course_id": 10, "phase": phase, "split": "train", "q": q, "y": label}
        for learner, q, label in [(1, 0.1, "E"), (2, 0.3, "G")]
        for phase in ("P1", "P2", "P3", "P4")
    ])
    assert validate_pool_b(snapshots)["rows"] == 8
    with pytest.raises(ValueError, match="P1–P4"):
        validate_pool_b(snapshots.iloc[:-1])
