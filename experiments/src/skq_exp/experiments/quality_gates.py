"""Đánh giá quality gate V2 từ evidence đã khai báo, không suy đoán pass.

Audit nhận mapping gate -> evidence ID đã được duyệt. Thiếu evidence, evidence bị
chặn và evidence fail là ba kết quả khác nhau; không gate nào được pass chỉ vì
không tìm thấy lỗi hoặc vì test khác đã pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import atomic_json, stable_hash
from ..reports.evidence import validate_evidence_manifest


def load_gate_specs(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("gates"), list):
        raise ValueError("Quality gate spec phải có schema_version=1 và gates")
    gate_ids = [gate.get("gate_id") for gate in payload["gates"]]
    if len(gate_ids) != len(set(gate_ids)) or any(not gate_id for gate_id in gate_ids):
        raise ValueError("Quality gate ID thiếu hoặc trùng")
    required = {"gate_id", "phase", "pass_condition", "evidence", "fail_action"}
    for gate in payload["gates"]:
        missing = required - set(gate)
        if missing:
            raise ValueError(f"{gate.get('gate_id')} thiếu trường: {sorted(missing)}")
    return payload["gates"]


def audit_quality_gates(
    artifact_root: str | Path,
    gate_specs: Sequence[Mapping[str, Any]],
    gate_evidence_map: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Audit từng gate độc lập và trả ledger đầy đủ, kể cả gate chưa chạy."""
    root = Path(artifact_root)
    rows = []
    for gate in gate_specs:
        gate_id = str(gate["gate_id"])
        evidence_ids = list(dict.fromkeys(gate_evidence_map.get(gate_id, ())))
        evidence_rows = []
        if not evidence_ids:
            status = "missing_evidence"
            reasons = ["Chưa khai báo evidence ID cho gate"]
        else:
            reasons = []
            for evidence_id in evidence_ids:
                path = root / "evidence" / "manifests" / f"{evidence_id}.json"
                if not path.exists():
                    evidence_rows.append({"evidence_id": evidence_id, "status": "missing"})
                    reasons.append(f"Thiếu manifest {evidence_id}")
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    validate_evidence_manifest(payload)
                    evidence_rows.append({
                        "evidence_id": evidence_id,
                        "status": payload["status"],
                        "manifest_id": payload["manifest_id"],
                    })
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                    evidence_rows.append({"evidence_id": evidence_id, "status": "invalid"})
                    reasons.append(f"{evidence_id}: {error}")
            statuses = {row["status"] for row in evidence_rows}
            if "invalid" in statuses:
                status = "invalid_evidence"
            elif "missing" in statuses:
                status = "missing_evidence"
            elif "fail" in statuses:
                status = "failed"
            elif "blocked" in statuses:
                status = "blocked"
            elif "partial" in statuses or "not_run" in statuses:
                status = "incomplete"
            elif statuses == {"pass"}:
                status = "pass"
            else:
                status = "incomplete"
        rows.append({
            "gate_id": gate_id,
            "phase": gate["phase"],
            "status": status,
            "evidence_ids": evidence_ids,
            "evidence_results": evidence_rows,
            "reasons": reasons,
            "pass_condition": gate["pass_condition"],
            "fail_action": gate["fail_action"],
        })
    body = {
        "schema_version": 1,
        "gate_results": rows,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "overall_status": "pass" if rows and all(row["status"] == "pass" for row in rows)
        else "not_ready",
    }
    return {**body, "audit_id": stable_hash(body)}


def write_gate_audit(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Gate audit đã tồn tại: {output}")
    atomic_json(output, payload)
    return output
