"""Run lỗi phải ở mẫu số và làm kiểm định thành not-estimable."""

from skq_exp.training.completeness import audit_run_completeness


def _expected():
    return [
        {"method_id": method, "selector_seed": seed, "phase": "P1"}
        for method in ("a", "b") for seed in (11, 29)
    ]


def test_failed_and_missing_cells_are_not_dropped() -> None:
    rows = [
        {**_expected()[0], "status": "success"},
        {**_expected()[1], "status": "timeout", "reason": "3h cap"},
        {**_expected()[2], "status": "success"},
    ]
    result = audit_run_completeness(rows, _expected())
    assert result["expected_denominator"] == 4
    assert result["successful_cells"] == 2
    assert result["completion_rate"] == 0.5
    assert result["failed_runs_kept_in_denominator"] is True
    assert result["inferential_ready"] is False
    assert result["inference_status"] == "not_estimable_incomplete"
    assert result["non_success_cells"][0]["status"] == "timeout"
    assert len(result["missing_cells"]) == 1


def test_complete_success_panel_is_ready() -> None:
    rows = [{**cell, "status": "success"} for cell in _expected()]
    result = audit_run_completeness(rows, _expected())
    assert result["inferential_ready"] is True
    assert result["completion_rate"] == 1.0
