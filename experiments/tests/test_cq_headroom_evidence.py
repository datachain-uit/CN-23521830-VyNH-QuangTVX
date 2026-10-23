"""Headroom CQ v1 chỉ nhận đủ bốn phase trong từng selector seed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compute_cq_v1_headroom import DEV_PHASES, _seed_ul, compute


def _rows():
    return [
        {
            "method_id": "c02_stratified_random",
            "learner_id": "xgb",
            "selector_seed": seed,
            "evaluation_split": phase,
            "status": "success",
            "macro_f1": 0.6 + 0.001 * index,
        }
        for seed in (11, 29, 47)
        for index, phase in enumerate(DEV_PHASES)
    ]


def test_headroom_aggregates_phase_inside_each_seed() -> None:
    values = _seed_ul(
        _rows(), method_id="c02_stratified_random", learner_id="xgb",
        expected_seeds=(11, 29, 47),
    )
    assert values == pytest.approx({11: 0.6015, 29: 0.6015, 47: 0.6015})


def test_headroom_rejects_missing_phase_instead_of_averaging_available_rows() -> None:
    with pytest.raises(ValueError, match="Thiếu/thừa dev phase"):
        _seed_ul(
            _rows()[:-1], method_id="c02_stratified_random", learner_id="xgb",
            expected_seeds=(11, 29, 47),
        )


def test_compute_headroom_uses_only_registered_fulltrain_and_c02(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configs"
    fulltrain = config_root / "cq_v1_m0_fulltrain_b_dev_natural_xgb_v1.json"
    c02_configs = sorted(config_root.glob("cq_v1_m0_headroom_c02_*_natural_v1.json"))
    fulltrain_experiment = "cq_v1_m0_fulltrain_b_dev_natural_xgb_v1"
    fulltrain_rows = [
        {
            "method_id": "c00_full_train", "learner_id": "xgb",
            "selector_seed": 0, "evaluation_split": phase,
            "status": "success", "macro_f1": 0.8,
        }
        for phase in DEV_PHASES
    ]
    fulltrain_ledger = tmp_path / fulltrain_experiment / "run_ledger.json"
    fulltrain_ledger.parent.mkdir(parents=True)
    fulltrain_ledger.write_text(json.dumps(fulltrain_rows), encoding="utf-8")

    scores = {0.005: 0.70, 0.01: 0.78, 0.02: 0.795, 0.05: 0.799}
    for config_path in c02_configs:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        experiment_id = payload["experiment_id"]
        ratio = float(payload["budget_ratio"])
        rows = _rows()
        for row in rows:
            row["macro_f1"] = scores[ratio]
        ledger = tmp_path / experiment_id / "run_ledger.json"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps(rows), encoding="utf-8")

    output = tmp_path / "headroom_decision.json"
    result = compute(tmp_path, fulltrain, c02_configs, output)
    assert result["rho_star"] == 0.01
    assert result["test_data_read"] is False
    assert output.is_file()
