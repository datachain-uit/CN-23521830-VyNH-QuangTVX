"""Các test này kiểm tra gate RAM và chẩn đoán số phép tính.

Ước lượng không thay thế benchmark thật. RAM dự báo vẫn là gate an toàn, còn số
phép tính chỉ để chẩn đoán và không được loại run theo Review 12–13.
"""

import pytest

from skq_exp.resources import (
    ResourceLimitError,
    enforce_preflight,
    estimate_selection_cost,
    operation_diagnostics,
)


def test_preflight_blocks_memory_heavy_run() -> None:
    estimate = estimate_selection_cost(
        "d04_global_rff_quadrature", 10_000_000, 100, 500_000,
        rff_components=1024, n_classes=2,
    )
    with pytest.raises(ResourceLimitError) as caught:
        enforce_preflight(estimate, {"max_ram_gb": 1, "max_estimated_operations": 1e20})
    assert caught.value.status == "predicted_oom"


def test_operation_estimate_is_diagnostic_not_a_gate() -> None:
    estimate = estimate_selection_cost(
        "a_craig_feature_space", 1_000, 20, 1_000, rff_components=1, n_classes=10
    )
    policy = {"max_ram_gb": 128, "max_estimated_operations": 1e6}
    enforce_preflight(estimate, policy)
    diagnostics = operation_diagnostics(estimate, policy)
    assert diagnostics["exceeds_legacy_reference"] is True
    assert diagnostics["decision_role"] == "diagnostic_only"


def test_kip_memory_estimate_includes_target_kernel_and_gradients() -> None:
    budget = 1_384
    estimate = estimate_selection_cost(
        "s_kip_tdbench", 27_676, 100, budget, rff_components=1, n_classes=2
    )
    # Riêng target-support kernel float64 đã là 10 * budget^2 * 8 byte;
    # estimate phải lớn hơn mức đó vì còn gradient và optimizer.
    raw_target_kernel = 10 * budget ** 2 * 8
    assert estimate.working_bytes > raw_target_kernel


def test_structured_estimate_uses_real_group_work_when_available() -> None:
    global_estimate = estimate_selection_cost(
        "p02_skq_coretab_xgb", 1_000_000, 50, 50_000,
        rff_components=256, n_classes=2,
    )
    grouped_estimate = estimate_selection_cost(
        "p02_skq_coretab_xgb", 1_000_000, 50, 50_000,
        rff_components=256, n_classes=2,
        structured_work_rows=2_000_000,
    )
    assert grouped_estimate.estimated_operations < global_estimate.estimated_operations


def test_d05_does_not_pay_rff_or_herding_estimate() -> None:
    estimate = estimate_selection_cost(
        "d05_equal_group_weight", 1_000_000, 50, 50_000,
        rff_components=256, n_classes=2,
    )
    assert estimate.rff_bytes == 0
    assert estimate.estimated_operations == 50_000_000


def test_streaming_rff_estimate_uses_bounded_group_storage() -> None:
    global_estimate = estimate_selection_cost(
        "p10_skq_mr_coretab_xgb", 5_000_000, 50, 250_000,
        rff_components=256, n_classes=2,
    )
    streaming = estimate_selection_cost(
        "p10_skq_mr_coretab_xgb", 5_000_000, 50, 250_000,
        rff_components=256, n_classes=2, streaming_rff_rows=10_000,
    )
    assert streaming.rff_bytes == 10_000 * 256 * 4
    assert streaming.working_bytes < global_estimate.working_bytes
