"""Manifest bằng chứng và Claim Ledger bất biến cho thực nghiệm V2.

Module chỉ lưu định danh, provenance và SHA-256. Artifact mức dòng vẫn ở vùng
private; manifest public không ghi đường dẫn máy hay nội dung dữ liệu. Một claim
empirical chỉ được đánh dấu ``supported`` khi mọi evidence tham chiếu hợp lệ và
các run_id của claim đều xuất hiện trong evidence.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import atomic_json, sha256_file, stable_hash


EVIDENCE_SCHEMA_VERSION = 1
CLAIM_SCHEMA_VERSION = 1
_EVIDENCE_ID = re.compile(r"^EV-[A-Z0-9][A-Z0-9_-]*-[0-9]{3}$")
_CLAIM_ID = re.compile(r"^CL-[A-Z0-9][A-Z0-9_-]*-[0-9]{3}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_EVIDENCE_STATUS = {"pass", "fail", "blocked", "partial", "not_run"}
_CLAIM_STATUS = {"supported", "descriptive_only", "blocked"}


def create_evidence_manifest(
    artifact_root: str | Path,
    *,
    evidence_id: str,
    requirement_ids: Sequence[str],
    code_commit: str,
    config_hash: str | None,
    split_hash: str | None,
    run_ids: Sequence[str],
    artifacts: Mapping[str, str | Path],
    visibility: str,
    producer_command: str,
    status: str,
) -> Path:
    """Tạo manifest evidence mới; không ghi đè evidence đã phát hành."""
    _validate_evidence_inputs(
        evidence_id=evidence_id,
        requirement_ids=requirement_ids,
        code_commit=code_commit,
        config_hash=config_hash,
        split_hash=split_hash,
        run_ids=run_ids,
        artifacts=artifacts,
        visibility=visibility,
        producer_command=producer_command,
        status=status,
    )
    hashes = {label: sha256_file(path) for label, path in artifacts.items()}
    body = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "requirement_ids": list(dict.fromkeys(requirement_ids)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "config_hash": config_hash,
        "split_hash": split_hash,
        "run_ids": list(dict.fromkeys(run_ids)),
        "artifact_sha256": hashes,
        "visibility": visibility,
        "producer_command": producer_command,
        "status": status,
    }
    payload = {**body, "manifest_id": stable_hash(body)}
    output = Path(artifact_root) / "evidence" / "manifests" / f"{evidence_id}.json"
    if output.exists():
        raise FileExistsError(f"Evidence đã tồn tại và bất biến: {output}")
    atomic_json(output, payload)
    return output


def validate_evidence_manifest(payload: Mapping[str, Any]) -> None:
    """Kiểm schema, ID và hash nội dung của một evidence manifest."""
    required = {
        "schema_version", "manifest_id", "evidence_id", "requirement_ids",
        "created_at_utc", "code_commit", "config_hash", "split_hash", "run_ids",
        "artifact_sha256", "visibility", "producer_command", "status",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Evidence manifest thiếu trường: {sorted(missing)}")
    body = {key: value for key, value in payload.items() if key != "manifest_id"}
    if payload["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise RuntimeError("Evidence schema_version không được hỗ trợ")
    if stable_hash(body) != payload["manifest_id"]:
        raise RuntimeError("Evidence manifest_id không khớp nội dung")
    _validate_evidence_inputs(
        evidence_id=str(payload["evidence_id"]),
        requirement_ids=payload["requirement_ids"],
        code_commit=str(payload["code_commit"]),
        config_hash=payload["config_hash"],
        split_hash=payload["split_hash"],
        run_ids=payload["run_ids"],
        artifacts={label: __file__ for label in payload["artifact_sha256"]},
        visibility=str(payload["visibility"]),
        producer_command=str(payload["producer_command"]),
        status=str(payload["status"]),
        check_artifact_paths=False,
    )
    if any(not _is_sha256(value) for value in payload["artifact_sha256"].values()):
        raise RuntimeError("Evidence chứa artifact hash không hợp lệ")


def create_claim_record(
    artifact_root: str | Path,
    *,
    claim_id: str,
    requirement_ids: Sequence[str],
    table_figure_cell: str,
    runs_long_row_keys: Sequence[Mapping[str, Any]],
    run_ids: Sequence[str],
    config_hash: str,
    split_hash: str,
    code_commit: str,
    evidence_ids: Sequence[str],
    raw_private_artifact_sha256: Mapping[str, str],
    allowed_claim: str,
    status: str,
) -> Path:
    """Tạo một claim record mới sau khi xác minh evidence tham chiếu."""
    if not _CLAIM_ID.fullmatch(claim_id):
        raise ValueError("claim_id phải theo mẫu CL-...-001")
    if not requirement_ids or not evidence_ids:
        raise ValueError("Claim phải có requirement_ids và evidence_ids")
    if not table_figure_cell.strip() or not allowed_claim.strip():
        raise ValueError("Claim phải chỉ rõ table/figure cell và phạm vi được phép")
    if status not in _CLAIM_STATUS:
        raise ValueError(f"Claim status không hợp lệ: {status}")
    if not all(_is_sha256(value) for value in (config_hash, split_hash)):
        raise ValueError("Claim cần config_hash và split_hash SHA-256")
    if any(not _SAFE_LABEL.fullmatch(label) or not _is_sha256(value)
           for label, value in raw_private_artifact_sha256.items()):
        raise ValueError("Private artifact chỉ được ghi label an toàn và SHA-256")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Claim không được lặp run_id")

    root = Path(artifact_root)
    evidence_payloads = []
    for evidence_id in evidence_ids:
        path = root / "evidence" / "manifests" / f"{evidence_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Claim thiếu evidence manifest: {evidence_id}")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        validate_evidence_manifest(evidence)
        evidence_payloads.append(evidence)

    body = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "claim_id": claim_id,
        "requirement_ids": list(dict.fromkeys(requirement_ids)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "table_figure_cell": table_figure_cell,
        "runs_long_row_keys": [dict(row) for row in runs_long_row_keys],
        "run_ids": list(run_ids),
        "provenance": {
            "config_hash": config_hash,
            "split_hash": split_hash,
            "code_commit": code_commit,
        },
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "raw_private_artifact_sha256": dict(raw_private_artifact_sha256),
        "allowed_claim": allowed_claim,
        "status": status,
    }
    _validate_claim_semantics(body, evidence_payloads)
    payload = {**body, "record_id": stable_hash(body)}
    output = root / "claims" / "records" / f"{claim_id}.json"
    if output.exists():
        raise FileExistsError(f"Claim đã tồn tại và bất biến: {output}")
    atomic_json(output, payload)
    return output


def audit_claim_ledger(artifact_root: str | Path) -> dict[str, Any]:
    """Audit toàn bộ claim; giữ từng finding riêng thay vì gộp thành một lỗi."""
    root = Path(artifact_root)
    findings: list[dict[str, str]] = []
    claim_paths = sorted((root / "claims" / "records").glob("*.json"))
    for path in claim_paths:
        try:
            claim = json.loads(path.read_text(encoding="utf-8"))
            _validate_claim_record_shape(claim)
            evidence_payloads: list[Mapping[str, Any]] = []
            for evidence_id in claim["evidence_ids"]:
                evidence_path = root / "evidence" / "manifests" / f"{evidence_id}.json"
                if not evidence_path.exists():
                    raise RuntimeError(f"thiếu evidence {evidence_id}")
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                validate_evidence_manifest(evidence)
                evidence_payloads.append(evidence)
            _validate_claim_semantics(claim, evidence_payloads)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            findings.append({"claim_file": path.name, "reason": str(error)})
    return {
        "schema_version": 1,
        "claims_checked": len(claim_paths),
        "findings": findings,
        "status": "pass" if not findings else "fail",
    }


def _validate_claim_record_shape(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "record_id", "claim_id", "requirement_ids",
        "created_at_utc", "table_figure_cell", "runs_long_row_keys", "run_ids",
        "provenance", "evidence_ids", "raw_private_artifact_sha256",
        "allowed_claim", "status",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Claim record thiếu trường: {sorted(missing)}")
    body = {key: value for key, value in payload.items() if key != "record_id"}
    if payload["schema_version"] != CLAIM_SCHEMA_VERSION:
        raise RuntimeError("Claim schema_version không được hỗ trợ")
    if stable_hash(body) != payload["record_id"]:
        raise RuntimeError("Claim record_id không khớp nội dung")
    if not _CLAIM_ID.fullmatch(str(payload["claim_id"])):
        raise RuntimeError("Claim ID không hợp lệ")


def _validate_claim_semantics(
    payload: Mapping[str, Any], evidence_payloads: Sequence[Mapping[str, Any]],
) -> None:
    """Áp cùng semantic gate khi tạo và khi audit record."""
    status = str(payload["status"])
    if status not in _CLAIM_STATUS:
        raise ValueError(f"Claim status không hợp lệ: {status}")
    if not payload["requirement_ids"] or not payload["evidence_ids"]:
        raise ValueError("Claim phải có requirement_ids và evidence_ids")
    if not str(payload["table_figure_cell"]).strip() or not str(payload["allowed_claim"]).strip():
        raise ValueError("Claim thiếu table/figure cell hoặc phạm vi được phép")
    provenance = payload["provenance"]
    if not _is_sha256(provenance.get("config_hash", "")) or not _is_sha256(
        provenance.get("split_hash", "")
    ):
        raise ValueError("Claim provenance hash không hợp lệ")
    if any(not _SAFE_LABEL.fullmatch(label) or not _is_sha256(value)
           for label, value in payload["raw_private_artifact_sha256"].items()):
        raise ValueError("Claim private artifact hash không hợp lệ")
    run_ids = list(payload["run_ids"])
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Claim không được lặp run_id")
    if status == "supported":
        if not run_ids or not payload["runs_long_row_keys"]:
            raise ValueError("Empirical claim supported cần run_ids và runs_long_row_keys")
        if any(evidence["status"] != "pass" for evidence in evidence_payloads):
            raise ValueError("Claim supported chỉ được tham chiếu evidence status=pass")
        evidenced_runs = {
            run_id for evidence in evidence_payloads for run_id in evidence["run_ids"]
        }
        missing_runs = set(run_ids) - evidenced_runs
        if missing_runs:
            raise ValueError(f"Claim có run chưa được evidence bao phủ: {sorted(missing_runs)}")


def _validate_evidence_inputs(
    *,
    evidence_id: str,
    requirement_ids: Sequence[str],
    code_commit: str,
    config_hash: str | None,
    split_hash: str | None,
    run_ids: Sequence[str],
    artifacts: Mapping[str, str | Path],
    visibility: str,
    producer_command: str,
    status: str,
    check_artifact_paths: bool = True,
) -> None:
    if not _EVIDENCE_ID.fullmatch(evidence_id):
        raise ValueError("evidence_id phải theo mẫu EV-...-001")
    if not requirement_ids or any(not str(item).strip() for item in requirement_ids):
        raise ValueError("Evidence phải có requirement_ids")
    if not code_commit.strip() or not producer_command.strip():
        raise ValueError("Evidence phải có code_commit và producer_command")
    if config_hash is not None and not _is_sha256(config_hash):
        raise ValueError("config_hash phải là SHA-256 hoặc null")
    if split_hash is not None and not _is_sha256(split_hash):
        raise ValueError("split_hash phải là SHA-256 hoặc null")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Evidence không được lặp run_id")
    if visibility not in {"public_manifest", "private"}:
        raise ValueError("visibility phải là public_manifest hoặc private")
    if status not in _EVIDENCE_STATUS:
        raise ValueError(f"Evidence status không hợp lệ: {status}")
    if status == "pass" and not artifacts:
        raise ValueError("Evidence pass phải có ít nhất một artifact hash")
    for label, path in artifacts.items():
        if not _SAFE_LABEL.fullmatch(label):
            raise ValueError(f"Artifact label không an toàn: {label}")
        if check_artifact_paths and not Path(path).is_file():
            raise FileNotFoundError(f"Không thấy artifact evidence: {path}")


def _is_sha256(value: str) -> bool:
    return bool(_SHA256.fullmatch(str(value)))
