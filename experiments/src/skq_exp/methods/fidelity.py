"""Kiểm toán độ trung thực của panel baseline trước M1 (quality gate G10).

Manifest tĩnh chỉ mô tả source, commit và adaptation đã đăng ký. Kết quả toy run
là evidence riêng sinh trên môi trường thực; thiếu/lỗi không được sửa thành
``passed`` trong manifest. Cách tách này ngăn việc tự khai báo đã tái hiện thành
công khi chưa chạy dependency ở Kaggle T4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .registry import METHOD_SPECS


MAIN_BASELINE_IDS = (
    "c01_uniform_random",
    "c02_stratified_random",
    "c03_phase_class_random",
    "n_gcoreset_benchmark",
    "b_moderate_ds_adapted",
    "b_ccs_oof_xgb_adapted",
    "b_dataoob_official",
    "b_kernel_thinning_classwise",
    "a_coretab_xgb_ova",
    "n_bdis_native",
)


def audit_fidelity_manifest(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    toy_evidence_path: str | Path | None = None,
    require_passed_toys: bool = False,
) -> dict[str, Any]:
    """Trả audit có cấu trúc; strict mode chỉ pass khi mọi toy run success."""
    manifest = _read_json(manifest_path)
    lock = _read_json(lock_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Fidelity manifest phải dùng schema_version=1")
    rows = manifest.get("methods")
    if not isinstance(rows, list):
        raise ValueError("Fidelity manifest thiếu methods list")
    by_id = {str(row.get("method_id")): row for row in rows}
    findings: list[dict[str, str]] = []

    for method_id in MAIN_BASELINE_IDS:
        row = by_id.get(method_id)
        if row is None:
            findings.append(_finding(method_id, "missing_manifest_row"))
            continue
        spec = METHOD_SPECS[method_id]
        for field, expected in (
            ("fidelity_tier", spec.fidelity_tier),
            ("task_contract", spec.task_contract),
            ("budget_contract", spec.budget_contract),
        ):
            if row.get(field) != expected:
                findings.append(_finding(
                    method_id, "registry_mismatch", f"{field}: {row.get(field)!r} != {expected!r}",
                ))
        source_key = row.get("source_lock_key")
        if source_key is not None:
            source = lock.get(source_key)
            if not isinstance(source, Mapping):
                findings.append(_finding(method_id, "missing_source_lock", str(source_key)))
            elif row.get("upstream_commit") != source.get("commit"):
                findings.append(_finding(method_id, "commit_mismatch", str(source_key)))
        if spec.fidelity_tier in {
            "official_source_adapter", "official_source_with_declared_patches",
            "benchmark_source_adapter",
        } and not row.get("adaptation_or_patch_notes"):
            findings.append(_finding(method_id, "missing_patch_declaration"))
        if not row.get("toy_test_id"):
            findings.append(_finding(method_id, "missing_toy_test_id"))

    unexpected = sorted(set(by_id) - set(MAIN_BASELINE_IDS))
    if unexpected:
        findings.append(_finding("manifest", "unexpected_methods", ",".join(unexpected)))

    toy_status = _toy_statuses(toy_evidence_path)
    if require_passed_toys:
        for method_id in MAIN_BASELINE_IDS:
            status = toy_status.get(method_id, "missing")
            if status != "success":
                findings.append(_finding(method_id, "toy_not_passed", status))

    return {
        "schema_version": 1,
        "manifest": str(Path(manifest_path)),
        "lock": str(Path(lock_path)),
        "toy_evidence": None if toy_evidence_path is None else str(Path(toy_evidence_path)),
        "require_passed_toys": bool(require_passed_toys),
        "method_count": len(MAIN_BASELINE_IDS),
        "toy_statuses": toy_status,
        "findings": findings,
        "status": "passed" if not findings else "failed",
    }


def _toy_statuses(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).is_file():
        return {}
    payload = _read_json(path)
    return {
        str(row.get("method_id")): str(row.get("status", "missing"))
        for row in payload.get("results", [])
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root phải là object: {path}")
    return payload


def _finding(method_id: str, kind: str, detail: str = "") -> dict[str, str]:
    return {"method_id": method_id, "kind": kind, "detail": detail}
