"""Paired course bootstrap và Holm cho K1–K5/K1c–K5c.

Hiệu ứng dương luôn có lợi cho A: UL(A)-UL(B), còn EC(B)-EC(A). Resampling dùng
course_root_id và cùng multiplicity cho mọi seed/phase trong một comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ULUnit:
    y_true: np.ndarray
    pred_a: np.ndarray
    pred_b: np.ndarray
    course_root_id: np.ndarray
    unit_id: str


@dataclass(frozen=True)
class ECUnit:
    error_a: np.ndarray
    error_b: np.ndarray
    course_root_id: np.ndarray
    unit_id: str


def paired_course_bootstrap_ul(
    units: tuple[ULUnit, ...],
    *,
    labels: tuple[Any, ...],
    n_boot: int = 2000,
    seed: int = 20261012,
    alpha: float = 0.05,
    non_inferiority_margin: float = 0.0,
    resampling_unit: str = "course_root_id",
) -> dict[str, Any]:
    """Bootstrap Delta_UL=MacroF1(A)-MacroF1(B), dương có lợi cho A."""
    if not units:
        raise ValueError("Cần ít nhất một UL unit")
    course_index = _common_course_index([
        np.asarray(unit.course_root_id) for unit in units
    ])
    cms = []
    for unit in units:
        _validate_equal_length(unit.y_true, unit.pred_a, unit.pred_b, unit.course_root_id)
        cms.append((
            _confusion_by_course(unit.y_true, unit.pred_a, unit.course_root_id, course_index, labels),
            _confusion_by_course(unit.y_true, unit.pred_b, unit.course_root_id, course_index, labels),
        ))

    def effect(counts: np.ndarray) -> float:
        values = []
        for cm_a, cm_b in cms:
            aggregate_a = np.tensordot(counts, cm_a, axes=1)
            aggregate_b = np.tensordot(counts, cm_b, axes=1)
            values.append(_macro_f1_from_confusion(aggregate_a) - _macro_f1_from_confusion(aggregate_b))
        return float(np.mean(values))

    return _bootstrap_result(
        effect,
        len(course_index),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        margin=non_inferiority_margin,
        effect_name="delta_ul_a_minus_b",
        unit_ids=[unit.unit_id for unit in units],
        resampling_unit=resampling_unit,
    )


def paired_course_bootstrap_ec(
    units: tuple[ECUnit, ...],
    *,
    n_boot: int = 2000,
    seed: int = 20261012,
    alpha: float = 0.05,
    non_inferiority_margin: float = 0.0,
    resampling_unit: str = "course_root_id",
) -> dict[str, Any]:
    """Bootstrap Delta_EC=MAE(B)-MAE(A), dương có lợi cho A."""
    if not units:
        raise ValueError("Cần ít nhất một EC unit")
    course_index = _common_course_index([
        np.asarray(unit.course_root_id) for unit in units
    ])
    position = {value: index for index, value in enumerate(course_index.tolist())}
    paired = []
    for unit in units:
        _validate_equal_length(unit.error_a, unit.error_b, unit.course_root_id)
        if len(np.unique(unit.course_root_id)) != len(unit.course_root_id):
            raise ValueError("EC unit phải có đúng một error mỗi course root")
        a = np.empty(len(course_index), dtype=float)
        b = np.empty(len(course_index), dtype=float)
        for course, error_a, error_b in zip(unit.course_root_id, unit.error_a, unit.error_b):
            index = position[course]
            a[index] = abs(float(error_a))
            b[index] = abs(float(error_b))
        paired.append((a, b))

    def effect(counts: np.ndarray) -> float:
        denominator = counts.sum()
        return float(np.mean([
            np.sum(counts * (error_b - error_a)) / denominator
            for error_a, error_b in paired
        ]))

    return _bootstrap_result(
        effect,
        len(course_index),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        margin=non_inferiority_margin,
        effect_name="delta_ec_b_minus_a",
        unit_ids=[unit.unit_id for unit in units],
        resampling_unit=resampling_unit,
    )


def holm_adjust(raw_p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm step-down cho một family; UL và EC phải gọi tách riêng."""
    if not raw_p_values:
        raise ValueError("Family Holm không được rỗng")
    for hypothesis, value in raw_p_values.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"p-value không hợp lệ: {hypothesis}={value}")
    ordered = sorted(raw_p_values, key=raw_p_values.get)
    size = len(ordered)
    running = 0.0
    result: dict[str, dict[str, Any]] = {}
    still_rejecting = True
    for rank, hypothesis in enumerate(ordered):
        raw = float(raw_p_values[hypothesis])
        adjusted = min(1.0, max(running, (size - rank) * raw))
        running = adjusted
        threshold = alpha / (size - rank)
        reject = bool(still_rejecting and raw <= threshold)
        if not reject:
            still_rejecting = False
        result[hypothesis] = {
            "raw_p": raw,
            "holm_adjusted_p": adjusted,
            "holm_threshold": float(threshold),
            "reject": reject,
        }
    return result


def _bootstrap_result(
    effect,
    n_courses: int,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    effect_name: str,
    unit_ids: list[str],
    resampling_unit: str,
) -> dict[str, Any]:
    if n_courses < 2 or n_boot <= 0 or not 0 < alpha < 1 or margin < 0:
        raise ValueError("Bootstrap cần >=2 course, n_boot>0, alpha trong (0,1), margin>=0")
    if resampling_unit not in {"course_root_id", "course_family_id"}:
        raise ValueError("resampling_unit phải là course_root_id hoặc course_family_id")
    point = effect(np.ones(n_courses, dtype=np.int64))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.integers(0, n_courses, size=n_courses)
        counts = np.bincount(sampled, minlength=n_courses)
        bootstrap[index] = effect(counts)
    lower_two, upper_two = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    lower_one = float(np.quantile(bootstrap, alpha))
    null_boundary = -float(margin)
    p_value = float((1 + np.count_nonzero(bootstrap <= null_boundary)) / (n_boot + 1))
    return {
        "schema_version": 1,
        "effect_name": effect_name,
        "effect_positive_favors": "A",
        "point_estimate": float(point),
        "two_sided_ci": [float(lower_two), float(upper_two)],
        "one_sided_lower_bound": lower_one,
        "raw_one_sided_p": p_value,
        "null_boundary": null_boundary,
        "non_inferiority_margin": float(margin),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "resampling_unit": resampling_unit,
        "course_count": int(n_courses),
        "unit_ids": unit_ids,
    }


def _common_course_index(arrays: list[np.ndarray]) -> np.ndarray:
    reference = np.unique(arrays[0])
    for values in arrays[1:]:
        if not np.array_equal(np.unique(values), reference):
            raise ValueError("Các unit/method phải có cùng course_root_id; không tự drop")
    return reference


def _confusion_by_course(y_true, y_pred, course, course_index, labels):
    labels = tuple(labels)
    label_position = {label: index for index, label in enumerate(labels)}
    course_position = {value: index for index, value in enumerate(course_index.tolist())}
    matrix = np.zeros((len(course_index), len(labels), len(labels)), dtype=np.int64)
    for truth, prediction, root in zip(y_true, y_pred, course):
        if truth not in label_position or prediction not in label_position:
            raise ValueError("Prediction/label ngoài class contract")
        matrix[course_position[root], label_position[truth], label_position[prediction]] += 1
    return matrix


def _macro_f1_from_confusion(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    keep = support > 0
    return float(np.mean(2 * true_positive[keep] / np.maximum(1.0, support[keep] + predicted[keep])))


def _validate_equal_length(*arrays) -> None:
    lengths = {len(np.asarray(array)) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("Các vector paired phải cùng chiều")
