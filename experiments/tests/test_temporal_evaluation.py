from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skq_exp.experiments.runner import _evaluation_splits
from skq_exp.config import ExperimentConfig
from skq_exp.methods.result import SelectionResult
from skq_exp.training import evaluate_fitted_learner, fit_selection_learner
from skq_exp.training import evaluator as evaluator_module


class _CountingLearner:
    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0
        self.predict_proba_calls = 0
        self.sample_weight = None

    def fit(self, X, y, sample_weight=None):
        del X, y
        self.fit_calls += 1
        self.sample_weight = np.asarray(sample_weight)
        return self

    def predict(self, X):
        self.predict_calls += 1
        return (np.asarray(X)[:, 0] > 0).astype(np.int64)

    def predict_proba(self, X):
        self.predict_proba_calls += 1
        positive = np.where(np.asarray(X)[:, 0] > 0, 0.8, 0.2)
        return np.column_stack([1.0 - positive, positive])


def test_full_course_quality_temporal_config_requires_freeze():
    root = Path(__file__).parents[1]
    config = ExperimentConfig.from_json(
        root / "configs" / "pilot_course_quality_med_temporal_seed11.json"
    )
    assert config.stage_id == "s4_temporal"
    assert config.test_locked is False
    assert config.requires_freeze_manifest is True
    assert config.dataset_ids == ("course_quality_med_v1",)


def test_course_quality_temporal_requires_four_splits():
    config = SimpleNamespace(stage_id="s4_temporal")
    data = {
        f"{prefix}_test_phase{phase}": np.zeros(2)
        for phase in range(1, 5)
        for prefix in ("X", "y")
    }
    assert _evaluation_splits(config, "course_quality_med_v1", data) == (
        "test_phase1", "test_phase2", "test_phase3", "test_phase4",
    )
    data.pop("X_test_phase4")
    with pytest.raises(RuntimeError, match="đủ test_phase1"):
        _evaluation_splits(config, "course_quality_med_v1", data)


def test_cq_v4_development_uses_all_four_dev_phases_and_never_test() -> None:
    config = SimpleNamespace(stage_id="s1_screen")
    data = {
        f"{prefix}_dev_phase{phase}": np.zeros(2)
        for phase in range(1, 5)
        for prefix in ("X", "y")
    }
    assert _evaluation_splits(config, "course_quality_v4", data) == (
        "dev_phase1", "dev_phase2", "dev_phase3", "dev_phase4",
    )
    data.pop("y_dev_phase2")
    with pytest.raises(RuntimeError, match="đủ dev P1-P4"):
        _evaluation_splits(config, "course_quality_v4", data)


def test_one_fitted_model_is_reused_for_four_temporal_splits(monkeypatch):
    learner = _CountingLearner()
    monkeypatch.setattr(
        evaluator_module, "build_learner", lambda *args, **kwargs: learner,
    )
    # Unit test không phụ thuộc thời gian khởi tạo CUDA/Torch của máy chạy test.
    monkeypatch.setattr(evaluator_module, "_peak_vram_bytes", lambda: 0)
    X_train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.asarray([0, 0, 1, 1], dtype=np.int64)
    selection = SelectionResult(
        indices=np.arange(4),
        weights=np.ones(4),
        requested_rows=4,
        realized_rows=4,
        budget_mode="exact_total",
        method_id="test_selector",
        diagnostics={},
        timings={"select": 0.01},
    )
    fitted = fit_selection_learner(
        selection, X_train, y_train, learner_id="lr",
    )
    costs = []
    for _phase in range(4):
        metrics, _predictions = evaluate_fitted_learner(
            fitted,
            X_train,
            y_train,
            evaluation_split_count=4,
        )
        costs.append(metrics["cost"])

    assert learner.fit_calls == 1
    assert learner.predict_calls == 4
    assert learner.predict_proba_calls == 4
    assert learner.sample_weight.mean() == pytest.approx(1.0)
    assert all(cost["fit_shared_across_evaluation_splits"] for cost in costs)
    assert all(cost["selection_shared_across_evaluation_splits"] for cost in costs)
    assert all(cost["fit_cost_allocation"] == "shared_not_additive" for cost in costs)
    assert all(cost["selection_cost_allocation"] == "shared_not_additive" for cost in costs)
    assert all(cost["evaluation_split_count"] == 4 for cost in costs)
    assert all(cost["peak_vram_bytes"] is None for cost in costs)
    assert all(cost["peak_vram_measurement"] == "not_measured" for cost in costs)
    assert all(
        cost["effective_sample_weight"]["mean"] == pytest.approx(1.0)
        for cost in costs
    )
