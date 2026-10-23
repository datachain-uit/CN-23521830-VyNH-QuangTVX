"""Dev-only CQ V4 A2/A4 and winner-quota-matched C03 preparation.

A2 reuses the unchanged SKQ-PC engine with one explicitly constant phase,
which is exactly class-only stratification. A4 reweights a successful source
selection, does not call QP again, and retains its already-paid source cost;
it isolates utility of weights, NOT measured speed of a QP-free selector.
Matched C03 copies realized source quotas rather than assuming proportional
allocation matches sqrt/floor. No winner is chosen and no test is opened here.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..base import BaseSelector
from ..contracts import exact_budget_size, validate_exact_selection
from ..result import SelectionResult
from .skq_pc import SKQPCSelector, _phase_class_keys


A2 = "a2_skq_pc_class_only"
A4 = "a4_skq_pc_uniform_weights"
C03_MATCHED = "c03_matched_skq_pc"
CQ_ABLATIONS = frozenset({A2, A4, C03_MATCHED})
SOURCES = frozenset({"skq_pc", "skq_pc_lrq_sq", "skq_pc_sqrt"})


def resolve_ablation_options(method_id: str, table: dict[str, dict]) -> dict:
    """Inherit source options; do not silently tune ablation-specific settings."""
    own = dict(table.get(method_id, {}))
    source = own.get("source_method")
    if source not in SOURCES:
        raise ValueError("CQ ablation requires an explicit registered source_method")
    if set(own) != {"source_method"}:
        raise ValueError("CQ ablation accepts only source_method; edit the source config explicitly")
    if source not in table:
        raise ValueError("CQ ablation source options must be declared in method_options")
    return {**table[source], "source_method": source}


class ClassOnlySKQPC(BaseSelector):
    """A2: retain geometry/herding/QP, collapse the phase partition explicitly."""

    def __init__(self, seed: int, options: dict[str, Any]):
        super().__init__(seed)
        source = options.get("source_method")
        if source not in SOURCES:
            raise ValueError("A2 requires explicit source_method")
        self.source_method = source
        self.engine = SKQPCSelector(
            source, variant="lrq_sq" if source == "skq_pc_lrq_sq" else "standard",
            seed=seed, n_components=int(options.get("n_components", 256)),
            max_cells_per_stratum=int(options.get("max_cells_per_stratum", 128)),
            bandwidth_multiplier=float(options.get("bandwidth_multiplier", 1.0)),
            query_alpha=float(options.get("query_alpha", 0.5)),
            allocation_policy=str(options.get("allocation_policy", "sqrt_floor" if source == "skq_pc_sqrt" else "proportional")),
            allocation_floor=int(options.get("allocation_floor", 20)),
        )

    def select(self, X_train, y_train, budget_ratio, *, phases=None, **kwargs):
        if phases is None or np.asarray(phases).shape != np.asarray(y_train).shape:
            return SelectionResult.failure(A2, "blocked", exact_budget_size(len(y_train), budget_ratio), "A2 requires aligned source phase metadata")
        result = self.engine.select(
            X_train, y_train, budget_ratio,
            phases=np.full(len(y_train), "all_phases", dtype=object), **kwargs,
        )
        result.method_id = A2
        result.diagnostics.update({
            "ablation_id": "A2", "source_method": self.source_method,
            "stratification": "class_only", "engine_constant_phase": "all_phases",
            "input_phase_count": int(len(np.unique(phases))),
            "phase_mass_preservation_required": False,
        })
        return result


def derive_phase_class_selection(
    method_id: str, source: SelectionResult, y: np.ndarray,
    phases: np.ndarray | None, budget_ratio: float, *, seed: int,
    resource_guard=None,
) -> SelectionResult:
    """A4 same rows, or C03 exact source quotas, with honest dependency timing."""
    y = np.asarray(y)
    requested = exact_budget_size(len(y), budget_ratio)
    if method_id not in {A4, C03_MATCHED}:
        raise ValueError("Unknown derived CQ method")
    if source.status != "success" or source.method_id not in SOURCES:
        return SelectionResult.failure(method_id, "blocked", requested, "Requires successful declared SKQ-PC candidate selection")
    if phases is None or np.asarray(phases).shape != y.shape:
        return SelectionResult.failure(method_id, "blocked", requested, "Requires aligned phase metadata")
    if source.realized_rows != requested or source.requested_rows != requested:
        return SelectionResult.failure(method_id, "blocked", requested, "Source budget differs from requested ablation budget")
    validate_exact_selection(source.indices, len(y), requested)
    started = time.perf_counter()
    keys, inverse, population = np.unique(_phase_class_keys(np.asarray(phases), y), return_inverse=True, return_counts=True)
    quotas = np.bincount(inverse[source.indices], minlength=len(keys))
    if np.any(quotas == 0):
        return SelectionResult.failure(method_id, "blocked", requested, "Source misses a phase-class stratum; cannot preserve all stratum masses")
    if method_id == A4:
        indices = source.indices.copy()
    else:
        rng = np.random.default_rng(seed)
        parts = []
        for group, quota in enumerate(quotas):
            if resource_guard is not None:
                resource_guard.check("cq_c03_matched")
            parts.append(rng.choice(np.flatnonzero(inverse == group), size=int(quota), replace=False))
        indices = np.concatenate(parts).astype(np.int64)
        indices = indices[rng.permutation(len(indices))]
    indices = validate_exact_selection(indices, len(y), requested)
    weights = population[inverse[indices]] / quotas[inverse[indices]]
    realized_mass = np.bincount(inverse[indices], weights=weights, minlength=len(keys))
    if not np.allclose(realized_mass, population, rtol=1e-10, atol=1e-8):
        raise RuntimeError("Derived CQ selection violated phase-class mass")
    elapsed = time.perf_counter() - started
    source_seconds = float(source.timings.get("total", source.timings.get("select", 0.0)))
    return SelectionResult(
        indices, weights, requested, requested, "exact_total", method_id,
        diagnostics={
            "ablation_id": "A4" if method_id == A4 else "C03_matched",
            "source_method": source.method_id,
            "same_indices": bool(np.array_equal(indices, source.indices)),
            "source_quota_matched": True, "source_selection_cost_shared": True,
            "strata": keys.tolist(), "stratum_capacities": population.tolist(),
            "stratum_quotas": quotas.tolist(),
            "allocation_policy": source.diagnostics.get("allocation_policy"),
            "allocation_floor": source.diagnostics.get("allocation_floor"),
            "stratum_mass_error_max": float(np.max(np.abs(realized_mass-population))),
            "qp_applied_in_derived_step": False,
            "source_qp_cost_already_incurred": True,
            "timing_interpretation": "utility_ablation_with_source_dependency_not_a_qp_free_selector_speed_measurement",
            "passes_over_train": float(source.diagnostics.get("passes_over_train", 0.0)) + 1.0,
            "passes_over_train_kind": "lower_bound",
            "passes_over_train_definition": "Source lower bound plus one logical phase-class metadata grouping; not a physical feature-pass count",
        },
        timings={"source_selection": source_seconds, "derived_step": elapsed, "total": source_seconds + elapsed},
    )
