"""Tạo quality-gate ledger từ spec và mapping evidence đã duyệt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skq_exp.experiments.quality_gates import (
    audit_quality_gates,
    load_gate_specs,
    write_gate_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--gate-spec", required=True)
    parser.add_argument("--evidence-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    specs = load_gate_specs(args.gate_spec)
    evidence_map = json.loads(Path(args.evidence_map).read_text(encoding="utf-8"))
    result = audit_quality_gates(args.artifact_root, specs, evidence_map)
    write_gate_audit(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
