"""Họ candidate và ablation SKQ-PC cho vòng chọn trên dev của CQ.

Trạng thái khoa học:

* ``skq_pc`` là candidate chuẩn được mô tả trong Review 12/10/2026.
* ``skq_pc_lrq_sq`` chỉ bổ sung loss OOF của Logistic Regression vào biểu diễn
  chọn mẫu; tuyệt đối không đọc dev/test khi tạo query feature.
* ``skq_pc_sqrt`` giữ pipeline chuẩn nhưng cấp quota phase×class tỷ lệ căn bậc
  hai với floor 20 theo QĐ20 của Review 22/10/2026.
* ``skq_pc_r`` giữ quota, cell và QP nhưng thay kernel herding bằng random;
  từ Review 22/10/2026 đây là ablation A3, không còn là candidate.
* ``x_coretab_mc_cell`` thay Gonzalez/Voronoi cell bằng leaf signature của
  CoreTab-XGB. Đây là candidate do nhóm suy luận, đang chờ giảng viên duyệt;
  không phải CoreTab native và không được đi vào freeze/test.

Cả bốn candidate dùng cùng contract: exact budget; tầng ngoài ``phase × class``;
largest remainder có tối thiểu một mẫu mỗi tầng khi khả thi; chọn dòng thật duy
nhất; simplex-QP theo tầng; và trọng số của mỗi tầng có tổng đúng bằng số dòng
gốc của tầng đó. Số cell thực tế là ``min(K, quota tầng, số dòng tầng)`` để
contract exact-budget luôn khả thi; quy tắc này được ghi vào diagnostics.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...resources import enforce_preflight, estimate_selection_cost
from ..base import BaseSelector
from ..contracts import BudgetInfeasibleError, exact_budget_size, validate_exact_selection
from ..controls import allocate_largest_remainder, allocate_stratum_quotas
from ..result import SelectionResult
from .herding import kernel_herding
from .rff import RBFRandomFeatures
from .simplex_qp import solve_simplex_mean_match


@dataclass(frozen=True)
class _StratumPlan:
    key: str
    rows: np.ndarray
    quota: int


class SKQPCSelector(BaseSelector):
    """Selector SKQ-PC dùng chung cho bốn candidate đã khai báo trước."""

    VALID_VARIANTS = {"standard", "lrq_sq", "random", "coretab_cell"}

    def __init__(
        self,
        method_id: str,
        *,
        variant: str,
        seed: int = 11,
        n_components: int = 256,
        max_cells_per_stratum: int = 128,
        bandwidth_multiplier: float = 1.0,
        query_alpha: float = 0.5,
        allocation_policy: str = "proportional",
        allocation_floor: int = 20,
    ) -> None:
        super().__init__(seed)
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"SKQ-PC variant không hợp lệ: {variant}")
        if n_components <= 0 or max_cells_per_stratum <= 0:
            raise ValueError("n_components và max_cells_per_stratum phải dương")
        if variant == "lrq_sq" and query_alpha != 0.5:
            raise ValueError("SKQ-PC + LRQ-SQ preregister query_alpha=0.5")
        self.method_id = method_id
        self.variant = variant
        self.n_components = int(n_components)
        self.max_cells_per_stratum = int(max_cells_per_stratum)
        self.bandwidth_multiplier = float(bandwidth_multiplier)
        self.query_alpha = float(query_alpha)
        self.allocation_policy = str(allocation_policy)
        self.allocation_floor = int(allocation_floor)
        if self.allocation_policy not in {"proportional", "sqrt_floor"}:
            raise ValueError("allocation_policy chỉ nhận proportional hoặc sqrt_floor")

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        budget_ratio: float,
        *,
        phases: np.ndarray | None = None,
        parent_ids: np.ndarray | None = None,
        row_ids: np.ndarray | None = None,
        query_features: np.ndarray | None = None,
        query_metadata: dict[str, Any] | None = None,
        resource_guard=None,
        **kwargs,
    ) -> SelectionResult:
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train)
        self.validate_input(X, y, budget_ratio)
        requested = exact_budget_size(len(y), budget_ratio)
        if phases is None:
            return SelectionResult.failure(
                self.method_id,
                "blocked",
                requested,
                "SKQ-PC cần phase_train của Pool B; không fallback sang class-only",
            )
        phases = np.asarray(phases)
        if phases.shape != y.shape:
            return SelectionResult.failure(
                self.method_id, "blocked", requested,
                "phase_train không thẳng hàng với y_train",
            )
        if self.variant == "coretab_cell":
            if parent_ids is None:
                return SelectionResult.failure(
                    self.method_id, "blocked", requested,
                    "SKQ-PC/CoreTab-cell cần leaf signature từ CoreTab-XGB",
                )
            parent_ids = np.asarray(parent_ids)
            if parent_ids.shape != y.shape:
                return SelectionResult.failure(
                    self.method_id, "blocked", requested,
                    "CoreTab leaf signature không thẳng hàng với y_train",
                )
        elif parent_ids is not None:
            raise ValueError("Chỉ SKQ-PC/CoreTab-cell được nhận parent_ids")

        row_ids = (
            np.arange(len(y), dtype=np.int64)
            if row_ids is None else np.asarray(row_ids, dtype=np.int64)
        )
        if row_ids.shape != y.shape or len(np.unique(row_ids)) != len(row_ids):
            raise ValueError("row_ids phải duy nhất và cùng chiều y_train")
        started = time.perf_counter()
        try:
            strata = self._build_strata(phases, y, requested)
        except BudgetInfeasibleError as error:
            return SelectionResult.failure(
                self.method_id, "budget_infeasible", requested, str(error)
            )

        estimate = estimate_selection_cost(
            self.method_id,
            len(y),
            X.shape[1],
            requested,
            rff_components=self.n_components,
            n_classes=len(np.unique(y)),
            structured_work_rows=float(sum(len(item.rows) * item.quota for item in strata)),
        )
        enforce_preflight(estimate, kwargs.get("resource_policy"))
        check = resource_guard.check if resource_guard is not None else None

        rff_started = time.perf_counter()
        rff = RBFRandomFeatures(
            n_components=self.n_components,
            bandwidth_multiplier=self.bandwidth_multiplier,
            seed=self.seed,
        )
        Z = rff.fit_transform(X, check=check)
        rff_scale = _root_mean_square_norm(Z)
        Z = Z / rff_scale
        query_scale = None
        if self.variant == "lrq_sq":
            if query_features is None:
                return SelectionResult.failure(
                    self.method_id, "blocked", requested,
                    "SKQ-PC + LRQ-SQ cần OOF-LR query feature train-only",
                )
            query = np.asarray(query_features, dtype=np.float32)
            if query.ndim == 1:
                query = query[:, None]
            if len(query) != len(y):
                raise ValueError("query_features phải có một dòng cho mỗi train row")
            query_scale = _root_mean_square_norm(query)
            Z = np.concatenate(
                [
                    np.sqrt(self.query_alpha) * Z,
                    np.sqrt(1.0 - self.query_alpha) * (query / query_scale),
                ],
                axis=1,
            )
        elif query_features is not None:
            raise ValueError("Chỉ SKQ-PC + LRQ-SQ được nhận query_features")
        rff_seconds = time.perf_counter() - rff_started

        selected_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        stratum_diagnostics: list[dict[str, Any]] = []
        qp_objectives: list[float] = []
        qp_iterations: list[int] = []
        qp_solvers: set[str] = set()
        qp_converged = True
        qp_constraints_valid = True
        cell_started = time.perf_counter()
        qp_seconds = 0.0
        for stratum_index, plan in enumerate(strata):
            if check is not None:
                check("skq_pc_stratum")
            if self.variant == "coretab_cell":
                assert parent_ids is not None
                cell_ids, cell_note = _coarsen_leaf_cells(
                    parent_ids[plan.rows], plan.quota
                )
                anchor_rows = np.empty(0, dtype=np.int64)
            else:
                cell_count = min(
                    self.max_cells_per_stratum, plan.quota, len(plan.rows)
                )
                cell_ids, anchor_local = _gonzalez_cells(
                    X[plan.rows],
                    cell_count,
                    row_ids[plan.rows],
                    self.seed + stratum_index,
                    check=check,
                )
                anchor_rows = row_ids[plan.rows[anchor_local]]
                cell_note = "gonzalez_voronoi"

            cells, cell_capacities = np.unique(cell_ids, return_counts=True)
            try:
                cell_quotas = allocate_largest_remainder(
                    cell_capacities, plan.quota, ensure_one=True
                )
            except BudgetInfeasibleError as error:
                return SelectionResult.failure(
                    self.method_id, "budget_infeasible", requested,
                    f"Tầng {plan.key} không cấp được quota cell: {error}",
                )
            stratum_selected: list[np.ndarray] = []
            for cell, quota in zip(cells, cell_quotas):
                local_pool = np.flatnonzero(cell_ids == cell)
                global_pool = plan.rows[local_pool]
                if self.variant == "random":
                    chosen_local = self.rng.choice(
                        len(global_pool), size=int(quota), replace=False
                    )
                else:
                    chosen_local = kernel_herding(
                        Z[global_pool], int(quota), row_ids[global_pool],
                        self.seed + stratum_index, check=check,
                    )
                stratum_selected.append(global_pool[chosen_local])
            picked = np.concatenate(stratum_selected).astype(np.int64)
            if len(picked) != plan.quota or len(np.unique(picked)) != len(picked):
                raise RuntimeError("SKQ-PC cell selection vi phạm exact/unique contract")

            qp_started = time.perf_counter()
            solution = solve_simplex_mean_match(
                Z[picked], Z[plan.rows].mean(axis=0), check=check
            )
            qp_seconds += time.perf_counter() - qp_started
            constraint_valid = bool(
                solution.sum_violation <= 1e-8
                and solution.min_weight >= -1e-10
            )
            qp_converged = qp_converged and bool(solution.converged)
            qp_constraints_valid = qp_constraints_valid and constraint_valid
            if not solution.converged or not constraint_valid:
                return SelectionResult.failure(
                    self.method_id, "failed", requested,
                    f"Simplex-QP tầng {plan.key} không hội tụ/vi phạm tolerance",
                )
            selected_parts.append(picked)
            weight_parts.append(float(len(plan.rows)) * solution.weights)
            qp_objectives.append(solution.objective)
            qp_iterations.append(solution.iterations)
            qp_solvers.add(solution.solver)
            stratum_diagnostics.append({
                "stratum": plan.key,
                "population": int(len(plan.rows)),
                "quota": int(plan.quota),
                "cell_source": cell_note,
                "cells": int(len(cells)),
                "cell_capacities": cell_capacities.astype(int).tolist(),
                "cell_quotas": cell_quotas.astype(int).tolist(),
                "anchor_row_ids": anchor_rows.astype(np.int64).tolist(),
                "qp_objective": float(solution.objective),
                "qp_converged": bool(solution.converged),
                "qp_constraint_valid": constraint_valid,
                "qp_sum_violation": float(solution.sum_violation),
                "qp_min_weight": float(solution.min_weight),
                "qp_solver": solution.solver,
                "qp_iterations": int(solution.iterations),
            })
        cell_and_qp_seconds = time.perf_counter() - cell_started

        indices = validate_exact_selection(
            np.concatenate(selected_parts), len(y), requested
        )
        weights = np.concatenate(weight_parts)
        keys = _phase_class_keys(phases, y)
        stratum_mass_errors = []
        for plan in strata:
            selected_mask = keys[indices] == plan.key
            stratum_mass_errors.append(
                abs(float(weights[selected_mask].sum()) - len(plan.rows))
                / max(1, len(plan.rows))
            )
        max_mass_error = float(max(stratum_mass_errors, default=0.0))
        total_mass_error = float(abs(weights.sum() - len(y)) / max(1, len(y)))
        if max_mass_error > 1e-8 or total_mass_error > 1e-8:
            raise RuntimeError("SKQ-PC không bảo toàn phase × class mass")

        return SelectionResult(
            indices=indices,
            weights=weights,
            requested_rows=requested,
            realized_rows=requested,
            budget_mode="exact_total",
            method_id=self.method_id,
            diagnostics={
                "algorithm_family": "SKQ-PC",
                "variant": self.variant,
                "selection_scope": "dev_only_pending_approval"
                if self.variant == "coretab_cell" else "approved_candidate",
                "phase_class_contract": True,
                "n_strata": int(len(strata)),
                "stratum_capacities": [int(len(item.rows)) for item in strata],
                "stratum_quotas": [int(item.quota) for item in strata],
                "allocation_policy": self.allocation_policy,
                "allocation_floor": self.allocation_floor,
                "strata": stratum_diagnostics,
                "max_cells_per_stratum_requested": self.max_cells_per_stratum,
                "cell_count_feasibility_rule": "min(K, stratum_quota, stratum_population)",
                "rff_components": self.n_components,
                "rff_sigma": rff.sigma_,
                "rff_landmark_row_ids": row_ids[rff.landmark_indices_].tolist(),
                "rff_block_scale": rff_scale,
                "query_alpha": self.query_alpha if self.variant == "lrq_sq" else None,
                "query_block_scale": query_scale,
                "query_metadata": query_metadata if self.variant == "lrq_sq" else None,
                "qp_objective_mean": float(np.mean(qp_objectives)),
                "qp_iterations_max": max(qp_iterations, default=0),
                "qp_solvers": sorted(qp_solvers),
                "qp_converged": qp_converged,
                "qp_constraint_valid": qp_constraints_valid,
                "max_stratum_mass_error": max_mass_error,
                "total_mass_error": total_mass_error,
                "passes_over_train": 2.0,
                "passes_over_train_kind": "lower_bound",
                "passes_over_train_definition": (
                    "Một lượt lập phase-class strata và một lượt biến đổi RFF; "
                    "không quy đổi các lần quét lặp trong Gonzalez, herding, QP "
                    "hoặc OOF query thành full-dataset pass."
                ),
                "class_coverage": float(
                    len(np.unique(y[indices])) / len(np.unique(y))
                ),
                "normalized_ess": float(
                    weights.sum() ** 2 / (len(weights) * np.sum(weights ** 2))
                ),
            },
            timings={
                "rff": rff_seconds,
                "cell_and_selection": max(0.0, cell_and_qp_seconds - qp_seconds),
                "qp": qp_seconds,
                "total": time.perf_counter() - started,
            },
        )

    def _build_strata(
        self, phases: np.ndarray, y: np.ndarray, requested: int,
    ) -> list[_StratumPlan]:
        keys = _phase_class_keys(phases, y)
        labels, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        quotas = allocate_stratum_quotas(
            counts,
            requested,
            policy=self.allocation_policy,
            floor=self.allocation_floor,
        )
        return [
            _StratumPlan(
                key=str(label),
                rows=np.flatnonzero(inverse == index).astype(np.int64),
                quota=int(quota),
            )
            for index, (label, quota) in enumerate(zip(labels, quotas))
        ]


def _phase_class_keys(phases: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{phase!r}\x1f{label!r}" for phase, label in zip(phases, y)],
        dtype=object,
    )


def _root_mean_square_norm(matrix: np.ndarray) -> float:
    value = float(
        np.sqrt(np.mean(np.sum(np.asarray(matrix, dtype=np.float64) ** 2, axis=1)))
    )
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Feature block có train RMS norm không hợp lệ")
    return value


def _tie_value(seed: int, row_id: int) -> int:
    raw = f"{seed}:{int(row_id)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _gonzalez_cells(
    X: np.ndarray,
    n_cells: int,
    row_ids: np.ndarray,
    seed: int,
    *,
    check=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo farthest-first anchors và Voronoi IDs trong đúng một tầng."""
    X = np.asarray(X, dtype=np.float64)
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if not 1 <= n_cells <= len(X):
        raise ValueError("n_cells phải nằm trong [1, số dòng tầng]")
    center = X.mean(axis=0)
    distance = np.sum((X - center) ** 2, axis=1)
    best = np.max(distance)
    tied = np.flatnonzero(np.isclose(distance, best, rtol=1e-12, atol=1e-12))
    tie_order = np.asarray([_tie_value(seed, value) for value in row_ids], dtype=np.uint64)
    first = int(tied[np.argmin(tie_order[tied])])
    anchors = [first]
    nearest = np.sum((X - X[first]) ** 2, axis=1)
    while len(anchors) < n_cells:
        if check is not None:
            check("skq_pc_gonzalez")
        nearest[np.asarray(anchors, dtype=np.int64)] = -np.inf
        best = np.max(nearest)
        tied = np.flatnonzero(np.isclose(nearest, best, rtol=1e-12, atol=1e-12))
        winner = int(tied[np.argmin(tie_order[tied])])
        anchors.append(winner)
        candidate_distance = np.sum((X - X[winner]) ** 2, axis=1)
        nearest = np.minimum(nearest, candidate_distance)
    anchors_array = np.asarray(anchors, dtype=np.int64)
    centers = X[anchors_array]
    distances = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (X @ centers.T)
    )
    return np.argmin(distances, axis=1).astype(np.int64), anchors_array


def _coarsen_leaf_cells(
    leaf_ids: np.ndarray, max_cells: int,
) -> tuple[np.ndarray, str]:
    """Gộp leaf cell nhỏ vào residual khi số leaf vượt quota của tầng."""
    leaf_ids = np.asarray(leaf_ids)
    labels, inverse, counts = np.unique(
        leaf_ids, return_inverse=True, return_counts=True
    )
    if len(labels) <= max_cells:
        return inverse.astype(np.int64), "coretab_leaf_signature"
    order = sorted(
        range(len(labels)),
        key=lambda index: (-int(counts[index]), repr(labels[index])),
    )
    keep = order[:max_cells - 1]
    output = np.full(len(leaf_ids), max_cells - 1, dtype=np.int64)
    for new_id, old_id in enumerate(keep):
        output[inverse == old_id] = new_id
    return output, "coretab_leaf_signature_with_residual_coarsening"
