"""File này tính metric từ prediction đã lưu, không train lại model.

Một confusion matrix được dùng lại cho metric tổng thể và từng nhãn nên chi phí
nhỏ. Metric cần xác suất hoặc phase chỉ chạy khi đầu vào tương ứng có mặt; giá trị
không xác định được ghi None kèm cảnh báo thay vì đổi thành 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, jaccard_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


def _divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def probability_is_valid(y_prob: np.ndarray | None, n_rows: int, n_classes: int) -> bool:
    if y_prob is None:
        return False
    values = np.asarray(y_prob)
    return (
        values.shape == (n_rows, n_classes)
        and np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    )


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[str]]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.asarray(labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])))
    warnings: list[str] = []
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": _safe_metric(cohen_kappa_score, y_true, y_pred),
        "quadratic_weighted_kappa": _safe_metric(
            cohen_kappa_score, y_true, y_pred, labels=labels, weights="quadratic"
        ),
    }
    for average in ("macro", "micro", "weighted"):
        result[f"precision_{average}"] = float(precision_score(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        ))
        result[f"recall_{average}"] = float(recall_score(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        ))
        result[f"f1_{average}"] = float(f1_score(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        ))
        result[f"jaccard_{average}"] = float(jaccard_score(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        ))
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    result["worst_class_f1"] = float(per_f1.min())
    result["worst_class_recall"] = float(per_recall.min())

    probability_metrics = [
        "roc_auc_macro", "roc_auc_weighted", "pr_auc_macro", "pr_auc_weighted",
        "log_loss", "brier_multiclass", "ece_15",
    ]
    if not probability_is_valid(y_prob, len(y_true), len(labels)):
        result.update({name: None for name in probability_metrics})
        warnings.append("Probability thiếu hoặc không hợp lệ; bỏ metric xác suất")
        return result, warnings
    y_prob = np.asarray(y_prob)
    one_hot = (y_true[:, None] == labels[None, :]).astype(np.float64)
    auc_values, ap_values = [], []
    for index, label in enumerate(labels):
        binary = (y_true == label).astype(int)
        if len(np.unique(binary)) < 2:
            auc_values.append(None)
            ap_values.append(None)
        else:
            auc_values.append(float(roc_auc_score(binary, y_prob[:, index])))
            ap_values.append(float(average_precision_score(binary, y_prob[:, index])))
    support = np.asarray([(y_true == label).sum() for label in labels], dtype=np.float64)
    result["roc_auc_macro"] = _mean_defined(auc_values)
    result["roc_auc_weighted"] = _weighted_defined(auc_values, support)
    result["pr_auc_macro"] = _mean_defined(ap_values)
    result["pr_auc_weighted"] = _weighted_defined(ap_values, support)
    result["log_loss"] = _safe_metric(log_loss, y_true, y_prob, labels=labels)
    result["brier_multiclass"] = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))
    result["ece_15"] = expected_calibration_error(y_true, y_prob, labels, 15)
    return result, warnings


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    y_true, y_pred, labels = np.asarray(y_true), np.asarray(y_pred), np.asarray(labels)
    valid_probability = probability_is_valid(y_prob, len(y_true), len(labels))
    rows = []
    for index, label in enumerate(labels):
        positive_true = y_true == label
        positive_pred = y_pred == label
        tp = int(np.sum(positive_true & positive_pred))
        fp = int(np.sum(~positive_true & positive_pred))
        tn = int(np.sum(~positive_true & ~positive_pred))
        fn = int(np.sum(positive_true & ~positive_pred))
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        specificity = _divide(tn, tn + fp)
        npv = _divide(tn, tn + fn)
        f1 = _divide(2 * tp, 2 * tp + fp + fn)
        f2 = _divide(5 * tp, 5 * tp + 4 * fn + fp)
        mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        row = {
            "label": _plain(label), "support": tp + fn,
            "prevalence": _divide(tp + fn, len(y_true)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity,
            "npv": npv, "f1": f1, "f2": f2,
            "fpr": _divide(fp, fp + tn), "fnr": _divide(fn, fn + tp),
            "mcc": _divide(tp * tn - fp * fn, mcc_den),
            "kappa": _safe_metric(cohen_kappa_score, positive_true, positive_pred),
            "roc_auc": None, "pr_auc": None,
        }
        if valid_probability and len(np.unique(positive_true)) == 2:
            row["roc_auc"] = float(roc_auc_score(positive_true, y_prob[:, index]))
            row["pr_auc"] = float(average_precision_score(positive_true, y_prob[:, index]))
        rows.append(row)
    return rows


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, labels: np.ndarray, n_bins: int,
) -> float:
    confidence = y_prob.max(axis=1)
    prediction = labels[y_prob.argmax(axis=1)]
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < n_bins - 1 else confidence <= upper)
        if mask.any():
            total += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(total)


def compute_temporal_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, horizons: np.ndarray, labels: np.ndarray,
) -> dict[str, Any]:
    rows = {}
    ordered = np.sort(np.unique(horizons))
    for horizon in ordered:
        mask = horizons == horizon
        rows[str(_plain(horizon))] = {
            "support": int(mask.sum()),
            "f1_macro": float(f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)),
        }
    values = np.asarray([rows[str(_plain(h))]["f1_macro"] for h in ordered])
    auc = None
    if len(ordered) > 1 and float(ordered[-1] - ordered[0]) > 0:
        auc = float(np.trapezoid(values, ordered) / (ordered[-1] - ordered[0]))
    return {
        "per_horizon": rows,
        "mean_horizon_f1": float(values.mean()),
        "worst_horizon_f1": float(values.min()),
        "auc_horizon_f1": auc,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    labels: np.ndarray | None = None,
    horizons: np.ndarray | None = None,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])))
    overall, warnings = compute_overall_metrics(y_true, y_pred, y_prob, labels)
    result = {
        "overall": overall,
        "per_label": compute_per_label_metrics(y_true, y_pred, y_prob, labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [_plain(label) for label in labels],
        "warnings": warnings,
    }
    if horizons is not None:
        result["temporal"] = compute_temporal_metrics(y_true, y_pred, np.asarray(horizons), labels)
    if cost is not None:
        result["cost"] = cost
    return result


def _safe_metric(function, *args, **kwargs) -> float | None:
    try:
        value = float(function(*args, **kwargs))
        return value if np.isfinite(value) else None
    except (ValueError, ZeroDivisionError):
        return None


def _mean_defined(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def _weighted_defined(values: list[float | None], weights: np.ndarray) -> float | None:
    pairs = [(value, weight) for value, weight in zip(values, weights) if value is not None]
    if not pairs:
        return None
    return float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs]))


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value

