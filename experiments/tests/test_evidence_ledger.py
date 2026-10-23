"""Evidence/claim phải bất biến, truy được run và không lộ path private."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skq_exp.reports.evidence import (
    audit_claim_ledger,
    create_claim_record,
    create_evidence_manifest,
)


SHA = "a" * 64


def _evidence(tmp_path: Path, *, status: str = "pass", run_ids=("run-1",)) -> Path:
    artifact = tmp_path / "private" / "metrics.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"macro_f1": 0.8}', encoding="utf-8")
    return create_evidence_manifest(
        tmp_path,
        evidence_id="EV-QD15-WEIGHTS-001",
        requirement_ids=("QD15", "G06"),
        code_commit="abc123",
        config_hash=SHA,
        split_hash="b" * 64,
        run_ids=run_ids,
        artifacts={"metrics.json": artifact},
        visibility="public_manifest",
        producer_command="pytest tests/test_weights.py",
        status=status,
    )


def test_evidence_contains_hash_not_private_path_and_is_immutable(tmp_path: Path) -> None:
    path = _evidence(tmp_path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "private" not in text
    assert payload["artifact_sha256"]["metrics.json"]
    with pytest.raises(FileExistsError):
        _evidence(tmp_path)


def test_supported_claim_requires_pass_evidence_and_run_coverage(tmp_path: Path) -> None:
    _evidence(tmp_path)
    claim = create_claim_record(
        tmp_path,
        claim_id="CL-TABLE2-UL-001",
        requirement_ids=("QD6",),
        table_figure_cell="Table 2 / SKQ-PC / UL",
        runs_long_row_keys=({"dataset_id": "cq_v1", "method_id": "skq_pc"},),
        run_ids=("run-1",),
        config_hash=SHA,
        split_hash="b" * 64,
        code_commit="abc123",
        evidence_ids=("EV-QD15-WEIGHTS-001",),
        raw_private_artifact_sha256={"predictions.parquet": "c" * 64},
        allowed_claim="Descriptive UL for the frozen confirmatory run",
        status="supported",
    )
    assert claim.exists()
    assert audit_claim_ledger(tmp_path)["status"] == "pass"


def test_supported_claim_rejects_uncovered_run(tmp_path: Path) -> None:
    _evidence(tmp_path)
    with pytest.raises(ValueError, match="chưa được evidence bao phủ"):
        create_claim_record(
            tmp_path,
            claim_id="CL-TABLE2-UL-002",
            requirement_ids=("QD6",),
            table_figure_cell="Table 2 / baseline / UL",
            runs_long_row_keys=({"dataset_id": "cq_v1"},),
            run_ids=("run-missing",),
            config_hash=SHA,
            split_hash="b" * 64,
            code_commit="abc123",
            evidence_ids=("EV-QD15-WEIGHTS-001",),
            raw_private_artifact_sha256={},
            allowed_claim="none",
            status="supported",
        )


def test_ledger_audit_detects_tampered_claim(tmp_path: Path) -> None:
    _evidence(tmp_path)
    claim = create_claim_record(
        tmp_path,
        claim_id="CL-TABLE2-UL-003",
        requirement_ids=("QD6",),
        table_figure_cell="Table 2 / SKQ-PC / UL",
        runs_long_row_keys=({"dataset_id": "cq_v1"},),
        run_ids=("run-1",),
        config_hash=SHA,
        split_hash="b" * 64,
        code_commit="abc123",
        evidence_ids=("EV-QD15-WEIGHTS-001",),
        raw_private_artifact_sha256={},
        allowed_claim="descriptive only",
        status="supported",
    )
    payload = json.loads(claim.read_text(encoding="utf-8"))
    payload["allowed_claim"] = "inflated claim"
    claim.write_text(json.dumps(payload), encoding="utf-8")
    result = audit_claim_ledger(tmp_path)
    assert result["status"] == "fail"
    assert len(result["findings"]) == 1
