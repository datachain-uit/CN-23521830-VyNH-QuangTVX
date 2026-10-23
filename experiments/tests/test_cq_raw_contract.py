"""Raw evidence contract phải giữ blocker thay vì suy luận từ Pool B."""

import json

from scripts.validate_cq_v1_raw_contract import validate


def test_empty_contract_remains_blocked(tmp_path) -> None:
    contract = {
        "group_unit_confirmation": {"relative_path": None},
        "fulltrain_a": {"relative_path": None},
        "D03_pre_imputation": {"relative_path": None},
        "D06_raw_events": {"relative_path": None},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    result = validate(path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["test_metrics_read"] is False
    assert all(
        item["status"] == "BLOCKED_INPUT_NOT_DECLARED"
        for item in result["checks"].values()
    )
