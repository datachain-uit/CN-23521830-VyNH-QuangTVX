"""Các test này kiểm tra selector giữ đúng budget, row ID và trọng số.

Dữ liệu nhỏ đủ để test chạy nhanh nhưng vẫn có nhiều lớp và parent group. Nếu một
thay đổi làm trùng index, mất lớp hoặc sai tổng khối lượng, test phải dừng ngay.
"""

import numpy as np

from skq_exp.data.toy import make_toy_data
from skq_exp.methods.controls import (
    PhaseClassStratifiedRandomSelector,
    StratifiedRandomSelector,
    UniformRandomSelector,
    allocate_stratum_quotas,
)
from skq_exp.methods.proposed.structured_kquad import StructuredKQuadSelector
from skq_exp.methods.proposed.herding import kernel_herding


def test_stratified_random_exact_and_deterministic() -> None:
    data = make_toy_data()
    first = StratifiedRandomSelector(11).select(data["X_train"], data["y_train"], 0.05)
    second = StratifiedRandomSelector(11).select(data["X_train"], data["y_train"], 0.05)
    assert first.status == "success"
    assert len(first.indices) == round(len(data["y_train"]) * 0.05)
    assert np.array_equal(first.indices, second.indices)
    assert len(np.unique(data["y_train"][first.indices])) == 3


def test_c01_uniform_random_is_exact_unique_and_deterministic() -> None:
    data = make_toy_data()
    first = UniformRandomSelector(11).select(data["X_train"], data["y_train"], 0.05)
    second = UniformRandomSelector(11).select(data["X_train"], data["y_train"], 0.05)
    assert first.status == "success"
    assert len(first.indices) == len(np.unique(first.indices))
    assert np.array_equal(first.indices, second.indices)


def test_c03_preserves_phase_class_mass_and_blocks_without_phase() -> None:
    data = make_toy_data()
    phases = np.resize(np.arange(4), len(data["y_train"]))
    result = PhaseClassStratifiedRandomSelector(11).select(
        data["X_train"], data["y_train"], 0.10, phases=phases,
    )
    assert result.status == "success"
    assert result.realized_rows == round(len(data["y_train"]) * 0.10)
    assert result.diagnostics["stratum_mass_error_max"] <= 1e-10
    blocked = PhaseClassStratifiedRandomSelector(11).select(
        data["X_train"], data["y_train"], 0.10,
    )
    assert blocked.status == "blocked"
    assert "không fallback" in blocked.diagnostics["reason"]


def test_sqrt_floor_allocation_is_exact_bounded_and_shared_by_c03() -> None:
    capacities = np.asarray([100, 25, 9, 4], dtype=np.int64)
    quotas = allocate_stratum_quotas(
        capacities, 40, policy="sqrt_floor", floor=2,
    )
    assert int(quotas.sum()) == 40
    assert np.all(quotas >= np.minimum(capacities, 2))
    assert np.all(quotas <= capacities)
    # Căn bậc hai làm quota/capacity của tầng hiếm lớn hơn tầng phổ biến.
    assert quotas[-1] / capacities[-1] > quotas[0] / capacities[0]

    data = make_toy_data()
    phases = np.resize(np.arange(4), len(data["y_train"]))
    result = PhaseClassStratifiedRandomSelector(
        11, allocation_policy="sqrt_floor", allocation_floor=1,
    ).select(data["X_train"], data["y_train"], 0.10, phases=phases)
    assert result.status == "success"
    assert result.diagnostics["allocation_policy"] == "sqrt_floor"


def test_structured_kquad_preserves_budget_and_mass() -> None:
    data = make_toy_data()
    selector = StructuredKQuadSelector(
        "p01_skq_coretab_dt", seed=11, n_components=64
    )
    result = selector.select(
        data["X_train"], data["y_train"], 0.05,
        parent_ids=data["parent_train"], row_ids=data["row_ids_train"],
    )
    assert result.status == "success"
    assert len(result.indices) == result.requested_rows
    assert len(np.unique(result.indices)) == len(result.indices)
    assert np.isclose(result.weights.sum(), len(data["y_train"]), atol=1e-5)
    assert result.diagnostics["class_coverage"] == 1.0
    assert result.diagnostics["max_mass_error"] < 1e-6


def test_structured_kquad_coarsens_microstrata_before_allocation() -> None:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(20, 5)).astype(np.float32)
    y = np.repeat([0, 1], 10)
    parent = np.arange(20, dtype=np.int64)  # một raw stratum cho mỗi dòng
    candidates = np.zeros(20, dtype=bool)
    candidates[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] = True
    result = StructuredKQuadSelector("p02_skq_coretab_xgb", seed=11, n_components=32).select(
        X, y, 0.4, parent_ids=parent, candidate_mask=candidates,
        row_ids=np.arange(100, 120, dtype=np.int64),
    )
    assert result.status == "success"
    assert result.diagnostics["raw_group_count"] == 20
    assert result.diagnostics["post_coarsening_group_count"] <= result.requested_rows
    assert min(result.diagnostics["group_quotas"]) >= 1
    assert np.isclose(result.weights.sum(), len(y), atol=1e-8)
    assert result.diagnostics["max_mass_error"] <= 1e-8


def test_structured_kquad_exact_policy_rejects_small_candidate_pool() -> None:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(20, 5)).astype(np.float32)
    y = np.repeat([0, 1], 10)
    candidates = np.zeros(20, dtype=bool)
    candidates[[0, 2, 4, 10, 12, 14]] = True
    result = StructuredKQuadSelector(
        "p02_skq_coretab_xgb", seed=11, n_components=16,
    ).select(X, y, 0.4, candidate_mask=candidates)
    assert result.status == "budget_infeasible"
    assert result.requested_rows == 8


def test_p03_caps_realized_size_at_bdis_candidate_pool() -> None:
    rng = np.random.default_rng(29)
    X = rng.normal(size=(20, 5)).astype(np.float32)
    y = np.repeat([0, 1], 10)
    candidates = np.zeros(20, dtype=bool)
    candidates[[0, 2, 4, 10, 12, 14]] = True
    result = StructuredKQuadSelector(
        "p03_skq_bdis_filtered", seed=11, n_components=16,
        budget_policy="cap_at_candidate_pool",
    ).select(X, y, 0.4, candidate_mask=candidates)
    assert result.status == "success"
    assert result.requested_rows == 8
    assert result.realized_rows == 6
    assert result.budget_mode == "candidate_pool_capped"
    assert result.diagnostics["parent_pool_limited"] is True
    assert result.diagnostics["requested_ratio"] == 0.4
    assert result.diagnostics["realized_ratio"] == 0.3
    assert np.isclose(result.weights.sum(), len(y), atol=1e-8)


def test_kernel_herding_uses_squared_distance_objective() -> None:
    Z = np.asarray([[5.0, 0.0], [1.0, 2.0], [-2.0, 1.0]], dtype=np.float64)
    target = Z.mean(axis=0)
    expected = int(np.argmin(np.sum((Z - target) ** 2, axis=1)))
    selected = kernel_herding(Z, 1, np.asarray([10, 11, 12]), seed=11)
    assert selected.tolist() == [expected]
