"""Audit Claim Ledger mà không đọc nội dung artifact private."""

from __future__ import annotations

import argparse
import json

from skq_exp.reports import audit_claim_ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    result = audit_claim_ledger(args.artifact_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
