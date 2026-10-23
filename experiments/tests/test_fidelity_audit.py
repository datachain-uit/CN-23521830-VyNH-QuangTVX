"""G10 chỉ pass strict khi source manifest và mọi toy reproduction đều đạt."""

import json
from pathlib import Path

from skq_exp.methods.fidelity import MAIN_BASELINE_IDS, audit_fidelity_manifest


ROOT = Path(__file__).parents[1]


def test_fidelity_manifest_structural_audit_passes() -> None:
    result = audit_fidelity_manifest(
        ROOT / "docs" / "method_fidelity_v2.json",
        ROOT / "external" / "official_repos.lock.json",
    )
    assert result["status"] == "passed"
    assert result["method_count"] == 10


def test_strict_audit_does_not_claim_unrun_toys(tmp_path: Path) -> None:
    result = audit_fidelity_manifest(
        ROOT / "docs" / "method_fidelity_v2.json",
        ROOT / "external" / "official_repos.lock.json",
        require_passed_toys=True,
    )
    assert result["status"] == "failed"
    assert len([item for item in result["findings"] if item["kind"] == "toy_not_passed"]) == 10

    evidence = tmp_path / "toy.json"
    evidence.write_text(json.dumps({
        "results": [
            {"method_id": method_id, "status": "success"}
            for method_id in MAIN_BASELINE_IDS
        ]
    }), encoding="utf-8")
    passed = audit_fidelity_manifest(
        ROOT / "docs" / "method_fidelity_v2.json",
        ROOT / "external" / "official_repos.lock.json",
        toy_evidence_path=evidence,
        require_passed_toys=True,
    )
    assert passed["status"] == "passed"
