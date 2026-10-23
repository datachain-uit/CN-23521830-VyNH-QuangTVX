"""Tạo evidence kiểm kê Kaggle trước calibration/run CQ v1."""

from __future__ import annotations

import argparse
import json

from skq_exp.experiments.environment import capture_environment, write_environment_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--session-limit-hours", type=float)
    parser.add_argument("--available-hours-per-week", type=float)
    parser.add_argument("--expected-gpu-model", default="T4")
    args = parser.parse_args()
    payload = capture_environment(
        provider="kaggle-free",
        accelerator_policy="gpu-t4",
        expected_gpu_model=args.expected_gpu_model,
        session_limit_hours=args.session_limit_hours,
        available_hours_per_week=args.available_hours_per_week,
    )
    output = write_environment_manifest(args.output, payload)
    print(json.dumps({"output": str(output), **payload}, ensure_ascii=False, indent=2))
    return 0 if payload["environment_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
