"""Quality gate phải giữ missing blocked và fail thành các trạng thái riêng."""

from __future__ import annotations

from pathlib import Path

from skq_exp.experiments.quality_gates import audit_quality_gates, load_gate_specs
from skq_exp.reports.evidence import create_evidence_manifest


def _make_evidence(root: Path, evidence_id: str, status: str) -> None:
    artifact = root / "inputs" / f"{evidence_id}.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(status, encoding="utf-8")
    create_evidence_manifest(
        root,
        evidence_id=evidence_id,
        requirement_ids=("fixture",),
        code_commit="abc123",
        config_hash=None,
        split_hash=None,
        run_ids=(),
        artifacts={f"{evidence_id}.txt": artifact},
        visibility="public_manifest",
        producer_command="fixture",
        status=status,
    )


def test_gate_audit_keeps_failure_classes_separate(tmp_path: Path) -> None:
    _make_evidence(tmp_path, "EV-GATE-PASS-001", "pass")
    _make_evidence(tmp_path, "EV-GATE-FAIL-001", "fail")
    _make_evidence(tmp_path, "EV-GATE-BLOCK-001", "blocked")
    specs = [
        {"gate_id": "G00", "phase": "x", "pass_condition": "x", "evidence": "x", "fail_action": "x"},
        {"gate_id": "G01", "phase": "x", "pass_condition": "x", "evidence": "x", "fail_action": "x"},
        {"gate_id": "G02", "phase": "x", "pass_condition": "x", "evidence": "x", "fail_action": "x"},
        {"gate_id": "G03", "phase": "x", "pass_condition": "x", "evidence": "x", "fail_action": "x"},
    ]
    result = audit_quality_gates(tmp_path, specs, {
        "G00": ("EV-GATE-PASS-001",),
        "G01": ("EV-GATE-FAIL-001",),
        "G02": ("EV-GATE-BLOCK-001",),
    })
    statuses = {row["gate_id"]: row["status"] for row in result["gate_results"]}
    assert statuses == {
        "G00": "pass", "G01": "failed", "G02": "blocked", "G03": "missing_evidence",
    }
    assert result["overall_status"] == "not_ready"


def test_committed_gate_spec_has_g00_through_g16() -> None:
    root = Path(__file__).parents[1]
    specs = load_gate_specs(root / "docs" / "quality_gates_v2.json")
    assert [gate["gate_id"] for gate in specs] == [f"G{i:02d}" for i in range(17)]
