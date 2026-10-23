"""Kiểm tra aggregate không biến artifact dở dang thành run thành công."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from skq_exp.reports.aggregate import aggregate_results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_without_metrics_is_interrupted(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_json(run / "manifest.json", {
        "dataset_id": "cq", "method_id": "c00_full_train",
        "learner_id": "cat", "status": "success",
    })

    output = aggregate_results(tmp_path)
    row = pd.read_csv(output).iloc[0]
    assert row["status"] == "interrupted"
    assert not bool(row["artifact_complete"])
    assert "metrics_overall.json" in row["reason"]


def test_failure_file_overrides_early_success_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_json(run / "manifest.json", {
        "dataset_id": "cq", "method_id": "c00_full_train",
        "learner_id": "cat", "status": "success",
    })
    _write_json(run / "evaluation_failure.json", {
        "status": "oom", "reason": "MemoryError",
    })

    output = aggregate_results(tmp_path)
    row = pd.read_csv(output).iloc[0]
    assert row["status"] == "oom"
    assert row["reason"] == "MemoryError"


def test_metrics_are_required_for_success(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_json(run / "manifest.json", {
        "dataset_id": "cq", "method_id": "c00_full_train",
        "learner_id": "xgb", "status": "success",
    })
    _write_json(run / "metrics_overall.json", {"f1_macro": 0.8, "mcc": 0.7})

    output = aggregate_results(tmp_path)
    row = pd.read_csv(output).iloc[0]
    assert row["status"] == "success"
    assert bool(row["artifact_complete"])

