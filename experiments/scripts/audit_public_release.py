"""CLI tạo privacy_audit.json trước khi push hoặc đóng gói public."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from skq_exp.reports.privacy import audit_public_release


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--archive", action="append", default=[])
    parser.add_argument("--output", default="artifacts/privacy/privacy_audit.json")
    args = parser.parse_args()
    evidence = audit_public_release(
        Path(args.project_root), archives=args.archive, output=args.output,
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
