"""Fit một learner một lần rồi đánh giá trên một hoặc nhiều split.

Dev/test do runner chọn rõ ràng. Với CourseQuality temporal, cùng model đã fit
được dùng lần lượt cho bốn test snapshot; không train lại bốn lần và không dùng
test để early stopping hay tuning.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning

from ..methods.result import SelectionResult
from ..methods.synthetic.result import GeneratedDatasetResult
from .learners import build_learner
from .metrics import compute_all_metrics
from .weights import (
    balanced_class_weights,
    effective_sample_weights,
    sample_weight_diagnostics,
)


@dataclass
class FittedLearner:
    model: Any
    labels: np.ndarray
    base_cost: dict[str, Any]


def fit_selection_learner(
    selection: SelectionResult,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    learner_id: str,
    model_seed: int = 42,
    learner_params: dict[str, Any] | None = None,
    X_inner_val: np.ndarray | None = None,
    y_inner_val: np.ndarray | None = None,
    max_threads: int = 4,
    class_weight_policy: str = "balanced",
    early_stopping: dict[str, Any] | None = None,
) -> FittedLearner:
    if selection.status != "success":
        raise RuntimeError(f"Không thể train từ selection status={selection.status}")
    selected_X = X_train[selection.indices]
    selected_y = y_train[selection.indices]
    class_weights = _class_weights(y_train, class_weight_policy)
    sample_weight = effective_sample_weights(
        selected_y, selection.weights, class_weights,
    )
    return _fit_learner(
        selected_X,
        selected_y,
        sample_weight,
        labels=np.unique(y_train),
        learner_id=learner_id,
        model_seed=model_seed,
        learner_params=learner_params,
        X_inner_val=X_inner_val,
        y_inner_val=y_inner_val,
        max_threads=max_threads,
        input_dim=X_train.shape[1],
        selection_seconds=selection.timings.get(
            "total", selection.timings.get("select"),
        ),
        requested_rows=selection.requested_rows,
        realized_rows=selection.realized_rows,
        full_train_rows=len(y_train),
        selection_breakdown=selection.timings,
        passes_over_train=selection.diagnostics.get("passes_over_train"),
        sample_weight_summary=sample_weight_diagnostics(sample_weight),
        effective_class_prior=_weighted_class_prior(
            selected_y, sample_weight, np.unique(y_train),
        ),
        class_weight_policy=class_weight_policy,
        early_stopping=early_stopping,
    )


def fit_generated_learner(
    generated: GeneratedDatasetResult,
    y_full_train: np.ndarray,
    *,
    learner_id: str,
    model_seed: int = 42,
    learner_params: dict[str, Any] | None = None,
    X_inner_val: np.ndarray | None = None,
    y_inner_val: np.ndarray | None = None,
    max_threads: int = 4,
    class_weight_policy: str = "balanced",
    early_stopping: dict[str, Any] | None = None,
) -> FittedLearner:
    """Fit learner trên synthetic table với class-weight từ full train."""
    if generated.status != "success":
        raise RuntimeError(f"Không thể train từ generated status={generated.status}")
    y_full_train = np.asarray(y_full_train)
    class_weights = _class_weights(y_full_train, class_weight_policy)
    sample_weight = effective_sample_weights(
        generated.y, generated.weights, class_weights,
    )
    return _fit_learner(
        generated.X,
        generated.y,
        sample_weight,
        labels=np.unique(y_full_train),
        learner_id=learner_id,
        model_seed=model_seed,
        learner_params=learner_params,
        X_inner_val=X_inner_val,
        y_inner_val=y_inner_val,
        max_threads=max_threads,
        input_dim=generated.X.shape[1],
        selection_seconds=generated.timings.get("total"),
        requested_rows=generated.requested_rows,
        realized_rows=generated.realized_rows,
        full_train_rows=len(y_full_train),
        selection_breakdown=generated.timings,
        passes_over_train=generated.diagnostics.get("passes_over_train"),
        sample_weight_summary=sample_weight_diagnostics(sample_weight),
        effective_class_prior=_weighted_class_prior(
            generated.y, sample_weight, np.unique(y_full_train),
        ),
        class_weight_policy=class_weight_policy,
        early_stopping=early_stopping,
    )


def evaluate_fitted_learner(
    fitted: FittedLearner,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    horizons: np.ndarray | None = None,
    evaluation_split_count: int = 1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Predict/metric cho một split mà không fit lại model."""
    process = psutil.Process()
    cpu_before = sum(process.cpu_times()[:2])
    rss_before = process.memory_info().rss
    io_before = _io_bytes(process)
    predict_started = time.perf_counter()
    y_pred = fitted.model.predict(X_eval)
    y_prob = (
        fitted.model.predict_proba(X_eval)
        if hasattr(fitted.model, "predict_proba")
        else None
    )
    predict_time = time.perf_counter() - predict_started
    cpu_after = sum(process.cpu_times()[:2])
    rss_after = process.memory_info().rss
    io_after = _io_bytes(process)
    peak_vram_bytes = _peak_vram_bytes()
    if peak_vram_bytes is not None and peak_vram_bytes <= 0:
        peak_vram_bytes = None

    cost = dict(fitted.base_cost)
    fit_io = cost.pop("io_bytes_fit", None)
    predict_io = None if io_before is None or io_after is None else io_after - io_before
    selection_seconds = cost.get("selection_seconds") or 0.0
    cost.update({
        "predict_seconds": predict_time,
        "total_seconds": selection_seconds + cost["fit_seconds"] + predict_time,
        "cpu_seconds_fit_predict": (
            cost.pop("cpu_seconds_fit") + cpu_after - cpu_before
        ),
        "rss_before_bytes": min(cost["rss_before_fit_bytes"], rss_before),
        "rss_after_bytes": rss_after,
        "peak_rss_bytes_lower_bound": max(
            cost.pop("rss_after_fit_bytes"), rss_before, rss_after,
        ),
        "io_bytes_fit_predict": (
            None if fit_io is None or predict_io is None else fit_io + predict_io
        ),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_measurement": (
            "torch_cuda_allocator" if peak_vram_bytes is not None else "not_measured"
        ),
        "evaluation_split_count": int(evaluation_split_count),
        "fit_shared_across_evaluation_splits": evaluation_split_count > 1,
        "fit_cost_allocation": "shared_not_additive",
        "selection_shared_across_evaluation_splits": evaluation_split_count > 1,
        "selection_cost_allocation": "shared_not_additive",
        "shared_selection_fit_seconds": selection_seconds + cost["fit_seconds"],
        "incremental_evaluation_seconds": predict_time,
    })
    cost.pop("rss_before_fit_bytes", None)
    metrics = compute_all_metrics(
        y_eval,
        y_pred,
        y_prob,
        labels=fitted.labels,
        horizons=horizons,
        cost=cost,
    )
    predictions = {"y_true": y_eval, "y_pred": y_pred, "y_prob": y_prob}
    return metrics, predictions


def _fit_learner(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    sample_weight: np.ndarray,
    *,
    labels: np.ndarray,
    learner_id: str,
    model_seed: int,
    learner_params: dict[str, Any] | None,
    X_inner_val: np.ndarray | None,
    y_inner_val: np.ndarray | None,
    max_threads: int,
    input_dim: int,
    selection_seconds: float | None,
    requested_rows: int,
    realized_rows: int,
    full_train_rows: int,
    selection_breakdown: dict[str, Any],
    passes_over_train: Any,
    sample_weight_summary: dict[str, float | int],
    effective_class_prior: dict[str, Any],
    class_weight_policy: str,
    early_stopping: dict[str, Any] | None,
) -> FittedLearner:
    if (X_inner_val is None) != (y_inner_val is None):
        raise ValueError("X_inner_val và y_inner_val phải cùng có hoặc cùng không")
    has_dev = X_inner_val is not None
    early_policy = dict(early_stopping or {})
    early_enabled = bool(early_policy.get("enabled", False))
    patience = int(early_policy.get("patience", 20))
    supports_early_stopping = learner_id in {"xgb", "cat", "mlp"}
    if early_enabled and early_policy.get("source") != "dev":
        raise ValueError("early stopping chỉ được dùng source=dev")
    if early_enabled and supports_early_stopping and not has_dev:
        raise ValueError(f"{learner_id} bật early stopping nhưng thiếu dev split")
    resolved_params = dict(learner_params or {})
    if early_enabled and has_dev:
        if learner_id == "xgb":
            resolved_params.setdefault("early_stopping_rounds", patience)
        elif learner_id == "cat":
            resolved_params.setdefault("use_best_model", True)
        elif learner_id == "mlp":
            resolved_params.setdefault("patience", patience)
    learner = build_learner(
        learner_id,
        seed=model_seed,
        input_dim=input_dim,
        n_classes=len(labels),
        params=resolved_params,
        max_threads=max_threads,
    )
    process = psutil.Process()
    cpu_before = sum(process.cpu_times()[:2])
    rss_before = process.memory_info().rss
    io_before = _io_bytes(process)
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if early_enabled and has_dev and learner_id == "xgb":
            learner.fit(
                X_fit,
                y_fit,
                sample_weight=sample_weight,
                eval_set=[(X_inner_val, y_inner_val)],
                verbose=False,
            )
        elif early_enabled and has_dev and learner_id == "cat":
            learner.fit(
                X_fit,
                y_fit,
                sample_weight=sample_weight,
                eval_set=(X_inner_val, y_inner_val),
                early_stopping_rounds=patience,
                verbose=False,
            )
        elif learner_id == "mlp":
            eval_set = (
                (X_inner_val, y_inner_val)
                if early_enabled and has_dev
                else None
            )
            learner.fit(X_fit, y_fit, sample_weight=sample_weight, eval_set=eval_set)
        else:
            learner.fit(X_fit, y_fit, sample_weight=sample_weight)
    convergence_warnings = [
        str(item.message) for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    for item in caught_warnings:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(str(item.message), item.category, stacklevel=2)
    fit_time = time.perf_counter() - fit_started
    cpu_after = sum(process.cpu_times()[:2])
    rss_after = process.memory_info().rss
    io_after = _io_bytes(process)
    base_cost = {
        "selection_seconds": selection_seconds,
        "fit_seconds": fit_time,
        "realized_rows": realized_rows,
        "requested_rows": requested_rows,
        "cpu_seconds_fit": cpu_after - cpu_before,
        "rss_before_fit_bytes": rss_before,
        "rss_after_fit_bytes": rss_after,
        "io_bytes_fit": (
            None if io_before is None or io_after is None else io_after - io_before
        ),
        "fit_rows_per_second": float(realized_rows / fit_time) if fit_time > 0 else None,
        "compression_ratio": float(realized_rows / full_train_rows),
        "selection_breakdown_seconds": selection_breakdown,
        "passes_over_train": passes_over_train,
        "effective_sample_weight": sample_weight_summary,
        "effective_class_prior": effective_class_prior,
        "class_weight_policy": class_weight_policy,
        "early_stopping_enabled": bool(early_enabled and supports_early_stopping),
        "early_stopping_source": "dev" if early_enabled and supports_early_stopping else None,
        "early_stopping_patience": patience if early_enabled and supports_early_stopping else None,
        "best_iteration": _best_iteration(learner),
        "fit_converged": (
            not convergence_warnings if learner_id == "lr" else None
        ),
        "optimizer_iterations": _optimizer_iterations(learner),
        "convergence_warnings": convergence_warnings,
    }
    return FittedLearner(model=learner, labels=labels, base_cost=base_cost)


def _class_weights(
    y_full_train: np.ndarray,
    policy: str,
) -> dict[int, float] | None:
    if policy == "natural":
        return None
    if policy == "balanced":
        return balanced_class_weights(y_full_train)
    raise ValueError("class_weight_policy chỉ nhận natural hoặc balanced")


def _weighted_class_prior(
    y: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Lưu prior learner thật sự thấy, căn theo labels để A8 tái lập được."""
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    labels = np.asarray(labels)
    if y.shape != weights.shape or weights.sum() <= 0:
        raise ValueError("Không thể tính effective class prior")
    probabilities = [float(weights[y == label].sum() / weights.sum()) for label in labels]
    return {
        "labels": [
            int(label) if isinstance(label, (int, np.integer)) else str(label)
            for label in labels
        ],
        "probabilities": probabilities,
    }


def _best_iteration(learner: Any) -> int | None:
    for attribute in ("best_iteration", "best_iteration_"):
        value = getattr(learner, attribute, None)
        if value is not None:
            return int(value)
    return None


def _optimizer_iterations(learner: Any) -> list[int] | None:
    value = getattr(learner, "n_iter_", None)
    if value is None:
        return None
    return [int(item) for item in np.asarray(value).reshape(-1)]


def _io_bytes(process: psutil.Process) -> int | None:
    try:
        counters = process.io_counters()
        return int(counters.read_bytes + counters.write_bytes)
    except (AttributeError, psutil.Error):
        return None


def _peak_vram_bytes() -> int | None:
    try:
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        return None
