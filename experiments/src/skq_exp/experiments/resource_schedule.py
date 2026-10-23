"""Tổng hợp QĐ16 thành dự báo lịch dev/test có truy vết.

Hàm chỉ cộng upper prediction bound từ phép đo thật. Chi phí learner/evaluation
phải được cung cấp bằng evidence riêng; nếu thiếu, kết quả là PARTIAL chứ không
được suy đoán PASS từ thời gian selector đơn thuần.
"""

from __future__ import annotations

from typing import Any


DEV_LIMIT_SECONDS = 4 * 24 * 60 * 60
TEST_LIMIT_SECONDS = 3 * 24 * 60 * 60


def compute_resource_schedule(
    calibration: dict[str, Any],
    *,
    fixed_methods: tuple[str, ...],
    candidate_methods: tuple[str, ...],
    selector_seeds: int = 3,
    dev_nonselector_seconds: float | None = None,
    test_nonselector_seconds: float | None = None,
) -> dict[str, Any]:
    """Tính lịch bảo thủ: dev chạy ba candidate, test chạy candidate chậm nhất."""
    if selector_seeds <= 0:
        raise ValueError("selector_seeds phải dương")
    methods = calibration.get("methods", {})
    expected = set(fixed_methods) | set(candidate_methods)
    missing = sorted(expected - set(methods))
    unexpected = sorted(set(methods) - expected)
    per_method: dict[str, Any] = {}
    unresolved = []
    for method_id in (*fixed_methods, *candidate_methods):
        item = methods.get(method_id, {})
        fit = item.get("fit", {})
        decision = item.get("decision", {})
        upper = fit.get("upper_prediction_seconds")
        state = decision.get("decision", "missing")
        usable = fit.get("fit_status") == "fitted" and upper is not None
        if not usable or state not in {"run_direct", "run_shared_shard_wrapper"}:
            unresolved.append(method_id)
        per_method[method_id] = {
            "fit_status": fit.get("fit_status", "missing"),
            "decision": state,
            "upper_prediction_seconds": None if upper is None else float(upper),
            "usable_for_schedule": usable,
        }

    if unresolved:
        dev_selector_seconds = None
        test_selector_seconds = None
    else:
        fixed_sum = sum(per_method[item]["upper_prediction_seconds"] for item in fixed_methods)
        candidate_sum = sum(
            per_method[item]["upper_prediction_seconds"] for item in candidate_methods
        )
        slowest_candidate = max(
            per_method[item]["upper_prediction_seconds"] for item in candidate_methods
        )
        dev_selector_seconds = selector_seeds * (fixed_sum + candidate_sum)
        test_selector_seconds = selector_seeds * (fixed_sum + slowest_candidate)

    overhead_complete = (
        dev_nonselector_seconds is not None and test_nonselector_seconds is not None
    )
    dev_total = (
        None if dev_selector_seconds is None or dev_nonselector_seconds is None
        else dev_selector_seconds + float(dev_nonselector_seconds)
    )
    test_total = (
        None if test_selector_seconds is None or test_nonselector_seconds is None
        else test_selector_seconds + float(test_nonselector_seconds)
    )
    if missing or unexpected or unresolved:
        status = "BLOCKED_INCOMPLETE_CALIBRATION"
    elif not overhead_complete:
        status = "PARTIAL_SELECTOR_ONLY"
    elif dev_total <= DEV_LIMIT_SECONDS and test_total <= TEST_LIMIT_SECONDS:
        status = "PASS"
    else:
        status = "FAIL_SCHEDULE"
    return {
        "schema_version": 1,
        "status": status,
        "test_data_read": False,
        "selector_seeds": selector_seeds,
        "planning_policy": {
            "dev": "all fixed selectors plus all three candidates",
            "test": "all fixed selectors plus the slowest candidate as conservative winner bound",
            "parallel_slots": 1,
            "prediction": "upper_prediction_seconds",
        },
        "expected_method_count": len(expected),
        "missing_methods": missing,
        "unexpected_methods": unexpected,
        "unresolved_methods": unresolved,
        "per_method": per_method,
        "dev": {
            "limit_seconds": DEV_LIMIT_SECONDS,
            "selector_seconds_upper": dev_selector_seconds,
            "nonselector_seconds_upper": dev_nonselector_seconds,
            "total_seconds_upper": dev_total,
            "within_limit": None if dev_total is None else dev_total <= DEV_LIMIT_SECONDS,
        },
        "test": {
            "limit_seconds": TEST_LIMIT_SECONDS,
            "selector_seconds_upper": test_selector_seconds,
            "nonselector_seconds_upper": test_nonselector_seconds,
            "total_seconds_upper": test_total,
            "within_limit": None if test_total is None else test_total <= TEST_LIMIT_SECONDS,
        },
        "overhead_evidence_complete": overhead_complete,
    }
