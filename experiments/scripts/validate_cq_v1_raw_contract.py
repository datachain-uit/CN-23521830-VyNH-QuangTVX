"""Kiểm tra đầu vào còn thiếu cho FullTrain-A, D03, D06 và group unit.

Script chỉ kiểm schema/provenance. Nó không biến Pool B/P4 thành FullTrain-A và
không tự suy luận course_root_id khi chưa có xác nhận từ nhóm dữ liệu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skq_exp.artifacts import atomic_json


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except ImportError:
        import pandas as pd

        return set(pd.read_parquet(path).columns)


def _check_parquet(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = spec.get("relative_path")
    if not relative:
        return {"status": "BLOCKED_INPUT_NOT_DECLARED", "path": None}
    path = (root / relative).resolve()
    if not path.is_file():
        return {"status": "BLOCKED_FILE_MISSING", "path": str(path)}
    columns = _parquet_columns(path)
    missing = sorted(set(spec.get("required_columns", ())) - columns)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    missing_manifest = list(spec.get("required_manifest_fields", ()))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_manifest = sorted(set(missing_manifest) - set(manifest))
    return {
        "status": "PASS" if not missing and not missing_manifest else "BLOCKED_SCHEMA",
        "path": str(path),
        "columns_present": len(columns),
        "missing_columns": missing,
        "manifest_path": str(manifest_path),
        "missing_manifest_fields": missing_manifest,
    }


def _check_group_confirmation(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    relative = spec.get("relative_path")
    if not relative:
        return {"status": "BLOCKED_INPUT_NOT_DECLARED", "path": None}
    path = (root / relative).resolve()
    if not path.is_file():
        return {"status": "BLOCKED_FILE_MISSING", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(set(spec.get("required_json_fields", ())) - set(payload))
    expected = spec.get("pass_value", {})
    mismatch = {key: {"expected": value, "actual": payload.get(key)} for key, value in expected.items() if payload.get(key) != value}
    return {
        "status": "PASS" if not missing and not mismatch else "BLOCKED_SCHEMA",
        "path": str(path),
        "missing_fields": missing,
        "value_mismatch": mismatch,
    }


def validate(contract_path: Path, source_root: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks = {
        "group_unit_confirmation": _check_group_confirmation(
            source_root, contract["group_unit_confirmation"],
        ),
        "fulltrain_a": _check_parquet(source_root, contract["fulltrain_a"]),
        "D03_pre_imputation": _check_parquet(source_root, contract["D03_pre_imputation"]),
        "D06_raw_events": _check_parquet(source_root, contract["D06_raw_events"]),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "BLOCKED",
        "test_metrics_read": False,
        "contract_path": str(contract_path),
        "source_root": str(source_root),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="docs/cq_v1/raw_input_contract_m0_v1.json")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", default="artifacts/cq_v1_m0_raw_contract_status.json")
    args = parser.parse_args()
    result = validate(Path(args.contract).resolve(), Path(args.source_root).resolve())
    atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
