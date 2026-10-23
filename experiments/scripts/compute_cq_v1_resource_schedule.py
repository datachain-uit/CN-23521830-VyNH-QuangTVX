"""Tạo bảng lịch QĐ16 từ calibration 10 selector + CoreTab OVA + ba candidate.

Script không đọc dev/test. Nếu chưa có evidence thời gian train/evaluate ngoài
selector, nó xuất PARTIAL_SELECTOR_ONLY và tuyệt đối không kết luận GO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skq_exp.artifacts import atomic_json
from skq_exp.experiments.resource_schedule import compute_resource_schedule
from scripts.run_cq_v1_resource_calibration import (
    ADAPTED_METHODS,
    CANDIDATE_METHODS,
    FIXED_METHODS,
)


def _optional_seconds(path: str | None) -> tuple[float | None, float | None, dict | None]:
    if path is None:
        return None, None, None
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return (
        float(payload["dev_nonselector_seconds_upper"]),
        float(payload["test_nonselector_seconds_upper"]),
        {"path": str(source), "payload": payload},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--overhead-evidence")
    parser.add_argument("--selector-seeds", type=int, default=3)
    parser.add_argument(
        "--output", default="artifacts/cq_v1_m0_resource_calibration/resource_schedule.json",
    )
    args = parser.parse_args()
    calibration_path = Path(args.calibration).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    dev_overhead, test_overhead, overhead_source = _optional_seconds(args.overhead_evidence)
    result = compute_resource_schedule(
        calibration,
        fixed_methods=FIXED_METHODS + ADAPTED_METHODS,
        candidate_methods=CANDIDATE_METHODS,
        selector_seeds=args.selector_seeds,
        dev_nonselector_seconds=dev_overhead,
        test_nonselector_seconds=test_overhead,
    )
    result["calibration_path"] = str(calibration_path)
    result["budget_ratio"] = calibration.get("budget_ratio")
    result["budget_source"] = calibration.get("budget_source")
    result["overhead_evidence"] = overhead_source
    output = Path(args.output).resolve()
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
