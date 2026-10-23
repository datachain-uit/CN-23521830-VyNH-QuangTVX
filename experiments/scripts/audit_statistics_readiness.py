"""Audit mẫu số method×seed×phase trước khi chạy bootstrap/Holm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skq_exp.training.completeness import audit_run_completeness


def _rows(path: str, field: str | None) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if field is not None:
        payload = payload[field]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path} phải chứa list object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--runs-field")
    parser.add_argument("--expected-field")
    parser.add_argument(
        "--key-fields", nargs="+", default=["method_id", "selector_seed", "phase"],
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    audit = audit_run_completeness(
        _rows(args.runs, args.runs_field),
        _rows(args.expected, args.expected_field),
        key_fields=args.key_fields,
    )
    rendered = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if audit["inferential_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
