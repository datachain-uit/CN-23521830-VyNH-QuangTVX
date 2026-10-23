"""Kiểm tra source/commit/patch và evidence toy của panel baseline G10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skq_exp.methods.fidelity import audit_fidelity_manifest


PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(PROJECT / "docs" / "method_fidelity_v2.json"),
    )
    parser.add_argument(
        "--lock", default=str(PROJECT / "external" / "official_repos.lock.json"),
    )
    parser.add_argument("--toy-evidence")
    parser.add_argument("--require-passed-toys", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    audit = audit_fidelity_manifest(
        args.manifest,
        args.lock,
        toy_evidence_path=args.toy_evidence,
        require_passed_toys=args.require_passed_toys,
    )
    rendered = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if audit["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
