"""Phân exact budget theo lớp rồi coarsen các ``parent x class`` micro-strata.

Đây là implementation của hợp đồng §9.3: class quota được cấp trước, sau đó các
micro-strata nhỏ/không có candidate được gộp xác định vào residual group. Mỗi
active group nhận ít nhất một dòng và không vượt candidate capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import BudgetInfeasibleError
from ..controls import allocate_largest_remainder


@dataclass(frozen=True)
class AllocationPlan:
    group_ids: np.ndarray
    groups: np.ndarray
    population_capacities: np.ndarray
    candidate_capacities: np.ndarray
    quotas: np.ndarray
    class_labels: np.ndarray
    class_quotas: np.ndarray
    diagnostics: dict


def make_group_ids(parent_ids: np.ndarray | None, y: np.ndarray) -> np.ndarray:
    """Mã hóa raw ``parent x class``; không dùng trực tiếp để cấp quota."""
    y = np.asarray(y)
    if parent_ids is None:
        _, encoded = np.unique(y, return_inverse=True)
        return encoded.astype(np.int64)
    parent_ids = np.asarray(parent_ids)
    if len(parent_ids) != len(y):
        raise ValueError("parent_ids phải cùng số dòng với y")
    pairs = np.asarray([f"{p!r}|{label!r}" for p, label in zip(parent_ids, y)], dtype=object)
    _, encoded = np.unique(pairs, return_inverse=True)
    return encoded.astype(np.int64)


def allocate_classwise(
    parent_ids: np.ndarray | None,
    y: np.ndarray,
    candidate_mask: np.ndarray,
    total: int,
) -> AllocationPlan:
    """Tạo coarsened groups và quota khả thi, xác định cho cùng input."""
    y = np.asarray(y)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if y.ndim != 1 or candidate_mask.shape != y.shape:
        raise ValueError("y và candidate_mask phải là mảng 1-D cùng chiều")
    if total <= 0 or total > int(candidate_mask.sum()):
        raise BudgetInfeasibleError("Candidate capacity không đủ exact budget")

    classes, class_inverse, class_populations = np.unique(
        y, return_inverse=True, return_counts=True
    )
    if total < len(classes):
        raise BudgetInfeasibleError("Budget nhỏ hơn số lớp nên không thể phủ lớp")
    class_candidates = np.bincount(
        class_inverse, weights=candidate_mask.astype(np.int64), minlength=len(classes)
    ).astype(np.int64)
    if np.any(class_candidates == 0):
        missing = classes[class_candidates == 0].tolist()
        raise BudgetInfeasibleError(f"Candidate pool không phủ các lớp: {missing}")
    class_quotas = _bounded_largest_remainder(
        class_populations, class_candidates, total, ensure_one=True
    )

    raw_parent = (
        np.zeros(len(y), dtype=np.int64) if parent_ids is None else np.asarray(parent_ids)
    )
    if len(raw_parent) != len(y):
        raise ValueError("parent_ids phải cùng số dòng với y")

    coarsened = np.full(len(y), -1, dtype=np.int64)
    group_classes: list[object] = []
    raw_group_count = 0
    next_group = 0
    for class_index, (label, class_quota) in enumerate(zip(classes, class_quotas)):
        class_rows = np.flatnonzero(class_inverse == class_index)
        raw_values = np.unique(raw_parent[class_rows])
        raw_group_count += len(raw_values)
        strata = []
        for raw_value in raw_values:
            rows = class_rows[raw_parent[class_rows] == raw_value]
            strata.append({
                "key": repr(raw_value),
                "rows": rows,
                "population": int(len(rows)),
                "candidates": int(candidate_mask[rows].sum()),
            })
        components = _coarsen_class(strata, int(class_quota))
        for component in components:
            rows = np.concatenate([item["rows"] for item in component])
            coarsened[rows] = next_group
            group_classes.append(label)
            next_group += 1

    if np.any(coarsened < 0):
        raise RuntimeError("Coarsening không gán hết train rows")
    groups, populations = np.unique(coarsened, return_counts=True)
    candidates = np.asarray([
        int(candidate_mask[coarsened == group].sum()) for group in groups
    ], dtype=np.int64)
    quotas = np.zeros(len(groups), dtype=np.int64)
    group_classes_array = np.asarray(group_classes)
    for label, class_quota in zip(classes, class_quotas):
        positions = np.flatnonzero(group_classes_array == label)
        quotas[positions] = _bounded_largest_remainder(
            populations[positions], candidates[positions], int(class_quota), ensure_one=True
        )

    if int(quotas.sum()) != total or np.any(quotas < 1) or np.any(quotas > candidates):
        raise BudgetInfeasibleError("Không tạo được group quota thỏa exact budget/candidate cap")

    probabilities = populations / populations.sum()
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    diagnostics = {
        "raw_group_count": int(raw_group_count),
        "post_coarsening_group_count": int(len(groups)),
        "group_merges": int(raw_group_count - len(groups)),
        "group_entropy": entropy,
        "effective_group_count": float(np.exp(entropy)),
        "singleton_share": float(np.mean(populations == 1)),
        "candidate_coverage": float(candidate_mask.mean()),
        "largest_mass_share": float(populations.max() / len(y)),
        "fraction_quota_le_3": float(np.mean(quotas <= 3)),
        "capacity_pressure": float(total / max(1, int(candidate_mask.sum()))),
        "class_labels": [str(value) for value in classes.tolist()],
        "class_populations": class_populations.astype(int).tolist(),
        "class_candidate_capacities": class_candidates.tolist(),
        "class_quotas": class_quotas.tolist(),
    }
    return AllocationPlan(
        group_ids=coarsened,
        groups=groups,
        population_capacities=populations.astype(np.int64),
        candidate_capacities=candidates,
        quotas=quotas,
        class_labels=classes,
        class_quotas=class_quotas,
        diagnostics=diagnostics,
    )


def _coarsen_class(strata: list[dict], class_quota: int) -> list[list[dict]]:
    if class_quota <= 0:
        raise BudgetInfeasibleError("Class quota phải dương")
    with_candidates = [item for item in strata if item["candidates"] > 0]
    if not with_candidates:
        raise BudgetInfeasibleError("Một lớp không có candidate")
    if len(strata) <= class_quota and len(with_candidates) == len(strata):
        return [[item] for item in sorted(strata, key=lambda item: item["key"])]

    # Giữ các strata lớn nhất; phần nhỏ và mọi strata không candidate đi vào residual.
    ordered = sorted(with_candidates, key=lambda item: (-item["population"], item["key"]))
    keep_count = min(class_quota - 1, len(ordered) - 1)
    kept_ids = {id(item) for item in ordered[:keep_count]}
    kept = [[item] for item in ordered[:keep_count]]
    residual = [item for item in strata if id(item) not in kept_ids]
    if not residual or sum(item["candidates"] for item in residual) <= 0:
        raise BudgetInfeasibleError("Residual group sau coarsening không có candidate")
    return kept + [sorted(residual, key=lambda item: item["key"])]


def _bounded_largest_remainder(
    populations: np.ndarray,
    candidate_capacities: np.ndarray,
    total: int,
    *,
    ensure_one: bool,
) -> np.ndarray:
    populations = np.asarray(populations, dtype=np.int64)
    candidate_capacities = np.asarray(candidate_capacities, dtype=np.int64)
    if len(populations) == 0 or len(populations) != len(candidate_capacities):
        raise BudgetInfeasibleError("Capacity arrays không hợp lệ")
    minimum = len(populations) if ensure_one else 0
    if total < minimum or int(candidate_capacities.sum()) < total:
        raise BudgetInfeasibleError("Quota không khả thi với candidate capacities")
    if ensure_one and np.any(candidate_capacities < 1):
        raise BudgetInfeasibleError("Active group không có candidate")
    quotas = allocate_largest_remainder(populations, total, ensure_one=ensure_one)
    quotas = np.minimum(quotas, candidate_capacities)
    left = int(total - quotas.sum())
    while left:
        eligible = np.flatnonzero(candidate_capacities > quotas)
        if len(eligible) == 0:
            raise BudgetInfeasibleError("Candidate capacity không đủ exact budget")
        scores = populations[eligible] / np.maximum(1, quotas[eligible] + 1)
        chosen = int(eligible[int(np.argmax(scores))])
        quotas[chosen] += 1
        left -= 1
    return quotas.astype(np.int64)


def allocate_groups(group_ids: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper cho caller cũ; chỉ dùng khi mọi dòng là candidate."""
    groups, counts = np.unique(group_ids, return_counts=True)
    quotas = _bounded_largest_remainder(counts, counts, total, ensure_one=total >= len(groups))
    return groups, counts, quotas
