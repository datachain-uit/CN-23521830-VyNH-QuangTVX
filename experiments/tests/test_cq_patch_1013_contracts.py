"""Ánh xạ bảy contract tối thiểu của ``cq_patch_1013`` vào mã production.

Các test giữ nguyên tên yêu cầu trong Review 13 để JUnit cho biết trực tiếp
contract nào đạt. Đây là test contract, không thay thế integration evidence CQ v1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skq_exp.data.cq_protocol import enrollment_table
from skq_exp.data.views import ColumnContract, build_feature_views
from skq_exp.experiments.headroom import choose_headroom_budget
from skq_exp.experiments.resource_calibration import (
    CALIBRATION_ROWS,
    CalibrationObservation,
    DeviceMetadata,
    fit_scaling_model,
)
from skq_exp.methods.adapted.coretab_ova import CoreTabXGBOVASelector
from skq_exp.methods.result import SelectionResult
from skq_exp.training.course_metrics import class_prevalence
from skq_exp.training.weights import effective_sample_weights


def test_xgb_depends_on_weight_scale_until_normalized() -> None:
    xgboost = pytest.importorskip("xgboost")
    X = np.arange(6, dtype=np.float32).reshape(-1, 1)
    y = np.asarray([0, 0, 0, 1, 1, 1])
    raw = np.asarray([1, 1, 1, 2, 2, 2], dtype=float)
    params = {
        "n_estimators": 1, "max_depth": 1, "learning_rate": 1.0,
        "reg_lambda": 10.0, "min_child_weight": 0, "gamma": 0,
        "tree_method": "hist", "n_jobs": 1, "random_state": 11,
    }
    first = xgboost.XGBClassifier(**params).fit(X, y, sample_weight=raw)
    scaled = xgboost.XGBClassifier(**params).fit(X, y, sample_weight=10 * raw)
    assert not np.allclose(first.predict_proba(X), scaled.predict_proba(X))
    assert np.allclose(
        effective_sample_weights(y, raw, None),
        effective_sample_weights(y, 10 * raw, None),
    )


def test_normalization_keeps_prior_and_mean_one() -> None:
    y = np.asarray([0, 0, 1, 2])
    raw = np.asarray([1.0, 3.0, 2.0, 4.0])
    normalized = effective_sample_weights(y, raw, None)
    assert normalized.mean() == pytest.approx(1.0)
    assert class_prevalence(y, raw) == pytest.approx(class_prevalence(y, normalized))


def test_selector_view_drops_id_columns() -> None:
    frame = pd.DataFrame({
        "learner_id": [1, 2], "course_id": [10, 20],
        "q": [0.1, 0.5], "y": [0, 2], "activity": [3.0, 7.0],
    })
    contracts = (
        ColumnContract("learner_id", "id", learner_allowed=True),
        ColumnContract("course_id", "id"),
        ColumnContract("q", "target_derived"),
        ColumnContract("y", "target"),
        ColumnContract("activity", "feature", learner_allowed=True, selector_allowed=True),
    )
    views = build_feature_views(frame, contracts, row_ids=np.asarray([0, 1]))
    assert views.selector_columns == ("activity",)


def test_enrollment_table_mean_over_chapters_and_labels_vary_within_course() -> None:
    chapters = pd.DataFrame({
        "learner_id": [1, 1, 2, 2], "course_id": [10, 10, 10, 10],
        "euclid_to_111": [0.1, 0.2, 0.5, 0.7],
    })
    table, _audit = enrollment_table(
        chapters, already_normalized=True, unopened_chapter_policy="included_as_rows",
    )
    assert table["q"].tolist() == pytest.approx([0.15, 0.6])
    assert table["y"].tolist() == ["E", "A"]


def test_headroom_rule() -> None:
    values = {
        ratio: {seed: value for seed, value in zip((11, 29, 47), scores)}
        for ratio, scores in {
            0.005: (0.76, 0.77, 0.765), 0.01: (0.785, 0.79, 0.787),
            0.02: (0.795, 0.796, 0.794), 0.05: (0.799, 0.80, 0.798),
        }.items()
    }
    assert choose_headroom_budget(full_train_b_ul=0.80, c02_dev_ul=values)["rho_star"] == 0.01


def test_scaling_gate_recovers_slope() -> None:
    device = DeviceMetadata(
        cpu="fixture", gpu=None, ram_bytes=16 * 1024**3, vram_bytes=None,
        max_threads=4, environment_id="cq-patch-contract",
    )
    observations = tuple(
        CalibrationObservation(
            selector_id="linear", rows=rows, budget_rows=rows // 100,
            replicate=replicate, warmup_seconds=0.1,
            steady_seconds=rows / 10_000, peak_rss_bytes=rows * 1024,
            peak_vram_bytes=None, device=device,
        )
        for rows in CALIBRATION_ROWS for replicate in (1, 2, 3)
    )
    fitted = fit_scaling_model(observations, target_rows=1_000_000)
    assert fitted["time_model"]["slope"] == pytest.approx(1.0, abs=1e-10)


def test_ovr_union_quota_disjoint_no_padding(monkeypatch) -> None:
    X = np.arange(90, dtype=np.float32).reshape(30, 3)
    y = np.repeat([0, 1, 2], 10)
    selector = CoreTabXGBOVASelector(Path("unused"), seed=11)

    def binary(_X, binary_y, _budget_ratio, _seed):
        positives = np.flatnonzero(binary_y == 1)[:2]
        return SelectionResult(
            indices=positives, weights=np.ones(len(positives)), requested_rows=9,
            realized_rows=len(positives), budget_mode="native_realized",
            method_id="n02_coretab_xgb_subset",
        )

    monkeypatch.setattr(selector, "_run_binary", binary)
    result = selector.select(X, y, 0.30)
    assert result.requested_rows == 9
    assert result.realized_rows == 6
    assert len(np.unique(result.indices)) == 6
    assert result.diagnostics["duplicates_removed"] == 0
