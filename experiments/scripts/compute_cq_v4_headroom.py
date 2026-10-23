"""Tính headroom CQ V4 với đúng 5 seed QĐ21, chỉ từ dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skq_exp.experiments.headroom import V4_HEADROOM_SEEDS
from scripts.compute_cq_v1_headroom import compute


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--fulltrain-config", default="configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json",
    )
    parser.add_argument("--c02-config", action="append", required=True)
    parser.add_argument(
        "--output", default="artifacts/cq_v4_m0_headroom/headroom_decision.json",
    )
    args = parser.parse_args()
    payload = compute(
        Path(args.artifact_root).resolve(),
        Path(args.fulltrain_config).resolve(),
        [Path(path).resolve() for path in args.c02_config],
        Path(args.output).resolve(),
        expected_seeds=V4_HEADROOM_SEEDS,
        protocol_id="cq_v4_m0_headroom_qd12_qd21_v1",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
