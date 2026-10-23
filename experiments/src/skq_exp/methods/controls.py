"""Đối chứng ngẫu nhiên phân tầng cho bảng so sánh.

Trạng thái: code nội bộ, không phải phương pháp đề xuất và không tuyên bố tái hiện
một paper cụ thể. Cơ chế: chia exact budget theo tỷ lệ lớp rồi lấy mẫu ngẫu nhiên
không hoàn lại. Loại đầu ra: tập con gồm các dòng thật, trọng số bằng nhau. Phương
pháp được đưa vào bài làm mốc cổ điển để kiểm tra SKQ có tốt hơn việc lấy mẫu ngẫu
nhiên nhưng vẫn bảo toàn phân bố nhãn hay không.
"""

from __future__ import annotations

import time

import numpy as np

from .base import BaseSelector
from .contracts import BudgetInfeasibleError, exact_budget_size, validate_exact_selection
from .result import SelectionResult


def allocate_largest_remainder(capacities: np.ndarray, total: int, ensure_one: bool = True) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64)
    if total > int(capacities.sum()):
        raise BudgetInfeasibleError("Budget lớn hơn tổng capacity")
    active = capacities > 0
    minimum = active.astype(np.int64) if ensure_one else np.zeros_like(capacities)
    if int(minimum.sum()) > total:
        raise BudgetInfeasibleError("Budget nhỏ hơn số nhóm/lớp cần phủ")
    remaining = total - int(minimum.sum())
    free = capacities - minimum
    if remaining == 0:
        return minimum
    shares = remaining * capacities / capacities.sum()
    extra = np.minimum(np.floor(shares).astype(np.int64), free)
    allocation = minimum + extra
    left = total - int(allocation.sum())
    remainders = shares - np.floor(shares)
    order = np.lexsort((np.arange(len(capacities)), -remainders))
    while left:
        progressed = False
        for index in order:
            if allocation[index] < capacities[index]:
                allocation[index] += 1
                left -= 1
                progressed = True
                if left == 0:
                    break
        if not progressed:
            raise BudgetInfeasibleError("Không thể phân bổ exact budget trong capacity")
    return allocation


def allocate_stratum_quotas(
    capacities: np.ndarray,
    total: int,
    *,
    policy: str = "proportional",
    floor: int = 20,
) -> np.ndarray:
    """Phân bổ exact budget theo hợp đồng QĐ20 của Review 22/10/2026.

    ``proportional`` giữ đối chứng legacy bằng largest remainder. ``sqrt_floor``
    dùng mục tiêu ``m_g ∝ sqrt(|P_g|)`` với cận dưới
    ``min(floor, |P_g|)`` và cận trên là capacity. Nếu tổng cận dưới lớn hơn
    budget thì trả lỗi thay vì âm thầm hạ floor, vì đó là một cấu hình khoa học
    khác với cấu hình đã đăng ký.
    """

    capacities = np.asarray(capacities, dtype=np.int64)
    if capacities.ndim != 1 or np.any(capacities < 0):
        raise ValueError("capacities phải là vector số nguyên không âm")
    if total < 0 or total > int(capacities.sum()):
        raise BudgetInfeasibleError("Budget nằm ngoài tổng capacity")
    if policy == "proportional":
        return allocate_largest_remainder(capacities, total, ensure_one=True)
    if policy != "sqrt_floor":
        raise ValueError(f"allocation policy không hợp lệ: {policy}")
    if floor <= 0:
        raise ValueError("allocation floor phải dương")

    lower = np.minimum(capacities, int(floor))
    if int(lower.sum()) > total:
        raise BudgetInfeasibleError(
            "Budget nhỏ hơn tổng floor phase×class của chính sách sqrt_floor"
        )
    if total == int(capacities.sum()):
        return capacities.copy()
    if total == int(lower.sum()):
        return lower.copy()

    weights = np.sqrt(capacities.astype(np.float64))
    low = 0.0
    high = max(
        (float(capacity) / weight for capacity, weight in zip(capacities, weights) if weight > 0),
        default=0.0,
    )
    for _ in range(100):
        scale = (low + high) / 2.0
        target = np.clip(scale * weights, lower, capacities)
        if float(target.sum()) < total:
            low = scale
        else:
            high = scale
    target = np.clip(high * weights, lower, capacities)
    allocation = np.floor(target + 1e-12).astype(np.int64)
    allocation = np.maximum(allocation, lower)
    allocation = np.minimum(allocation, capacities)
    left = total - int(allocation.sum())
    remainders = target - allocation
    order = np.lexsort((np.arange(len(capacities)), -remainders))
    while left > 0:
        progressed = False
        for index in order:
            if allocation[index] < capacities[index]:
                allocation[index] += 1
                left -= 1
                progressed = True
                if left == 0:
                    break
        if not progressed:
            raise BudgetInfeasibleError("Không thể phân bổ exact sqrt-floor budget")
    if int(allocation.sum()) != total:
        raise RuntimeError("sqrt-floor allocator không bảo toàn exact budget")
    return allocation


class StratifiedRandomSelector(BaseSelector):
    method_id = "c02_stratified_random"

    def select(self, X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float, **kwargs) -> SelectionResult:
        self.validate_input(X_train, y_train, budget_ratio)
        started = time.perf_counter()
        requested = exact_budget_size(len(y_train), budget_ratio)
        labels, counts = np.unique(y_train, return_counts=True)
        quotas = allocate_largest_remainder(counts, requested, ensure_one=True)
        chosen = []
        for label, quota in zip(labels, quotas):
            pool = np.flatnonzero(y_train == label)
            chosen.extend(self.rng.choice(pool, size=int(quota), replace=False).tolist())
        indices = validate_exact_selection(np.asarray(chosen), len(y_train), requested)
        self.rng.shuffle(indices)
        return SelectionResult(
            indices=indices,
            weights=np.ones(requested, dtype=np.float64),
            requested_rows=requested,
            realized_rows=requested,
            budget_mode="exact_total",
            method_id=self.method_id,
            diagnostics={
                "labels": labels.tolist(),
                "class_capacities": counts.tolist(),
                "class_quotas": quotas.tolist(),
                "passes_over_train": 1.0,
                "passes_over_train_kind": "exact_logical",
                "passes_over_train_definition": (
                    "Một lượt logic để lập strata theo lớp; thao tác lấy mẫu "
                    "không được tính như một lượt đọc toàn bộ feature matrix."
                ),
            },
            timings={"select": time.perf_counter() - started},
        )


class UniformRandomSelector(BaseSelector):
    """C01: lấy mẫu đều không hoàn lại với exact budget."""

    method_id = "c01_uniform_random"

    def select(
        self, X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float, **kwargs,
    ) -> SelectionResult:
        del kwargs
        self.validate_input(X_train, y_train, budget_ratio)
        started = time.perf_counter()
        requested = exact_budget_size(len(y_train), budget_ratio)
        indices = validate_exact_selection(
            self.rng.choice(len(y_train), size=requested, replace=False),
            len(y_train), requested,
        )
        return SelectionResult(
            indices=indices,
            weights=np.ones(requested, dtype=np.float64),
            requested_rows=requested,
            realized_rows=requested,
            budget_mode="exact_total",
            method_id=self.method_id,
            diagnostics={
                "sampling": "uniform_without_replacement",
                "passes_over_train": 0.0,
                "passes_over_train_kind": "exact_logical",
                "passes_over_train_definition": (
                    "Selector sinh index trực tiếp và không quét feature matrix."
                ),
            },
            timings={"select": time.perf_counter() - started},
        )


class PhaseClassStratifiedRandomSelector(BaseSelector):
    """C03: random theo phase × class, weight bảo toàn mass từng tầng."""

    method_id = "c03_phase_class_random"

    def __init__(
        self,
        seed: int = 11,
        *,
        allocation_policy: str = "proportional",
        allocation_floor: int = 20,
    ) -> None:
        super().__init__(seed)
        self.allocation_policy = allocation_policy
        self.allocation_floor = int(allocation_floor)

    def select(
        self, X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float, **kwargs,
    ) -> SelectionResult:
        self.validate_input(X_train, y_train, budget_ratio)
        requested = exact_budget_size(len(y_train), budget_ratio)
        phases = kwargs.get("phases")
        if phases is None:
            return SelectionResult.failure(
                self.method_id, "blocked", requested,
                "C03 cần phase_train của Pool B; không fallback sang C02",
            )
        phases = np.asarray(phases)
        if phases.shape != np.asarray(y_train).shape:
            return SelectionResult.failure(
                self.method_id, "blocked", requested,
                "phase_train không thẳng hàng với y_train",
            )
        started = time.perf_counter()
        keys = np.asarray([f"{phase}\x1f{label}" for phase, label in zip(phases, y_train)])
        strata, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        try:
            quotas = allocate_stratum_quotas(
                counts,
                requested,
                policy=self.allocation_policy,
                floor=self.allocation_floor,
            )
        except BudgetInfeasibleError as error:
            return SelectionResult.failure(
                self.method_id, "budget_infeasible", requested, str(error),
            )
        selected: list[int] = []
        weights: list[float] = []
        for stratum_index, (capacity, quota) in enumerate(zip(counts, quotas)):
            pool = np.flatnonzero(inverse == stratum_index)
            chosen = self.rng.choice(pool, size=int(quota), replace=False)
            selected.extend(chosen.tolist())
            weights.extend([float(capacity / quota)] * int(quota))
        indices = np.asarray(selected, dtype=np.int64)
        method_weights = np.asarray(weights, dtype=np.float64)
        order = self.rng.permutation(len(indices))
        indices = validate_exact_selection(indices[order], len(y_train), requested)
        method_weights = method_weights[order]
        realized_mass = np.bincount(
            inverse[indices], weights=method_weights, minlength=len(strata),
        )
        if not np.allclose(realized_mass, counts):
            raise RuntimeError("C03 không bảo toàn mass phase × class")
        return SelectionResult(
            indices=indices,
            weights=method_weights,
            requested_rows=requested,
            realized_rows=requested,
            budget_mode="exact_total",
            method_id=self.method_id,
            diagnostics={
                "strata": strata.tolist(),
                "stratum_capacities": counts.tolist(),
                "stratum_quotas": quotas.tolist(),
                "allocation_policy": self.allocation_policy,
                "allocation_floor": self.allocation_floor,
                "stratum_mass_error_max": float(np.max(np.abs(realized_mass - counts))),
                "passes_over_train": 1.0,
                "passes_over_train_kind": "exact_logical",
                "passes_over_train_definition": (
                    "Một lượt logic để lập strata phase-class; thao tác lấy mẫu "
                    "trong strata không được tính như lượt đọc toàn bộ feature matrix."
                ),
            },
            timings={"select": time.perf_counter() - started},
        )
