"""Lịch QĐ16 chỉ PASS khi đủ panel và đủ overhead ngoài selector."""

from skq_exp.experiments.resource_schedule import compute_resource_schedule


FIXED = tuple(f"f{index}" for index in range(10))
CANDIDATES = tuple(f"p{index}" for index in range(3))


def _calibration():
    return {
        "methods": {
            method: {
                "fit": {"fit_status": "fitted", "upper_prediction_seconds": 100.0},
                "decision": {"decision": "run_direct"},
            }
            for method in (*FIXED, *CANDIDATES)
        }
    }


def test_schedule_is_partial_without_nonselector_overhead() -> None:
    result = compute_resource_schedule(
        _calibration(), fixed_methods=FIXED, candidate_methods=CANDIDATES,
    )
    assert result["status"] == "PARTIAL_SELECTOR_ONLY"
    assert result["dev"]["selector_seconds_upper"] == 3900.0
    assert result["test"]["selector_seconds_upper"] == 3300.0


def test_schedule_passes_only_with_complete_overhead() -> None:
    result = compute_resource_schedule(
        _calibration(), fixed_methods=FIXED, candidate_methods=CANDIDATES,
        dev_nonselector_seconds=1000.0, test_nonselector_seconds=1000.0,
    )
    assert result["status"] == "PASS"
    assert result["dev"]["within_limit"] is True
    assert result["test"]["within_limit"] is True


def test_schedule_blocks_when_one_method_lacks_fit() -> None:
    calibration = _calibration()
    calibration["methods"]["f3"]["fit"] = {"fit_status": "needs_more_calibration"}
    result = compute_resource_schedule(
        calibration, fixed_methods=FIXED, candidate_methods=CANDIDATES,
    )
    assert result["status"] == "BLOCKED_INCOMPLETE_CALIBRATION"
    assert result["unresolved_methods"] == ["f3"]
