"""ID/target-derived không được trở thành tọa độ của selector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skq_exp.data.views import ColumnContract, build_feature_views


CONTRACTS = (
    ColumnContract("enrollment_id", "id", learner_allowed=True),
    ColumnContract("course_id", "id"),
    ColumnContract("q", "target_derived"),
    ColumnContract("y", "target"),
    ColumnContract("event_count", "feature", learner_allowed=True, selector_allowed=True),
    ColumnContract("duration", "feature", learner_allowed=True, selector_allowed=True),
    ColumnContract("school_type", "metadata", learner_allowed=True),
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "enrollment_id": [100, 101, 102],
        "course_id": [10, 10, 20],
        "q": [0.1, 0.7, 0.4],
        "y": [0, 2, 1],
        "event_count": [2, 9, 5],
        "duration": [1.0, 4.0, 2.0],
        "school_type": [0, 1, 1],
    })


def test_selector_and_learner_views_have_explicit_semantics() -> None:
    views = build_feature_views(_frame(), CONTRACTS, row_ids=np.array([7, 8, 9]))
    assert views.selector_columns == ("event_count", "duration")
    assert views.learner_columns == (
        "enrollment_id", "event_count", "duration", "school_type",
    )
    assert np.array_equal(views.row_ids, [7, 8, 9])
    assert len(views.contract_hash) == 64


def test_permuting_ids_does_not_change_selector_view() -> None:
    original = _frame()
    permuted = original.copy()
    permuted["enrollment_id"] = [999, 555, 777]
    permuted["course_id"] = [30, 40, 50]
    first = build_feature_views(original, CONTRACTS, row_ids=np.array([7, 8, 9]))
    second = build_feature_views(permuted, CONTRACTS, row_ids=np.array([7, 8, 9]))
    pd.testing.assert_frame_equal(first.selector, second.selector)


def test_target_derived_column_cannot_be_mislabeled_as_feature() -> None:
    invalid = tuple(
        ColumnContract("q", "feature", learner_allowed=True, selector_allowed=True)
        if contract.name == "q" else contract
        for contract in CONTRACTS
    )
    with pytest.raises(ValueError, match="target/target-derived"):
        build_feature_views(_frame(), invalid, row_ids=np.array([7, 8, 9]))


def test_selected_indices_address_same_rows_in_learner_view() -> None:
    views = build_feature_views(_frame(), CONTRACTS, row_ids=np.array([7, 8, 9]))
    selected = np.array([2, 0])
    assert views.row_ids[selected].tolist() == [9, 7]
    assert views.learner.iloc[selected]["event_count"].tolist() == [5, 2]
