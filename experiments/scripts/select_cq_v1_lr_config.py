"""Chọn LR M0 từ tối đa tám cấu hình FullTrain-B chỉ trên dev.

Điều kiện hợp lệ: đủ bốn phase, mọi fit hội tụ và có CQ endpoint. Xếp hạng chính
theo UL dev, sau đó Macro-F1, EC thấp hơn và thời gian fit. Script không đọc test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.config import ExperimentConfig


EXPECTED_EVALS = {f"dev_phase{index}" for index in range(1, 5)}


def _single_endpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    successful = [item for item in payload.get("runs", []) if item.get("status") == "success"]
    if len(successful) != 1:
        raise ValueError(f"{path} cần đúng một endpoint thành công, có {len(successful)}")
    return successful[0]


def _collect(config_path: Path) -> dict[str, Any]:
    config = ExperimentConfig.from_json(config_path)
    if tuple(config.learner_ids) != ("lr",) or tuple(config.method_ids) != ("c00_full_train",):
        raise ValueError(f"Config tuning phải chỉ có FullTrain-B/LR: {config_path}")
    root = config.paths.artifact_root / "experiments" / config.experiment_id
    rows = []
    convergence = []
    fit_seconds = []
    for manifest_path in root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = manifest.get("evaluation_split")
        if evaluation not in EXPECTED_EVALS:
            continue
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics_overall.json"
        cost_path = run_dir / "cost.json"
        if not metrics_path.is_file() or not cost_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        rows.append({
            "evaluation_split": evaluation,
            "macro_f1": float(metrics["f1_macro"]),
            "mcc": float(metrics["mcc"]),
        })
        convergence.append(bool(cost.get("fit_converged")))
        fit_seconds.append(float(cost["fit_seconds"]))
    observed = {item["evaluation_split"] for item in rows}
    endpoint_path = root / "cq_dev_endpoints.json"
    endpoint = _single_endpoint(endpoint_path) if endpoint_path.is_file() else None
    eligible = observed == EXPECTED_EVALS and all(convergence) and endpoint is not None
    pcc = endpoint.get("pcc", {}) if endpoint else {}
    return {
        "experiment_id": config.experiment_id,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "class_weight_policy": config.class_weight_policy,
        "learner_params": (config.learner_params or {}).get("lr", {}),
        "observed_evaluation_splits": sorted(observed),
        "all_four_dev_phases": observed == EXPECTED_EVALS,
        "fit_converged_all": len(convergence) == 4 and all(convergence),
        "fit_seconds_shared": None if not fit_seconds else float(max(fit_seconds)),
        "macro_f1_mean": None if not rows else float(np.mean([item["macro_f1"] for item in rows])),
        "mcc_mean": None if not rows else float(np.mean([item["mcc"] for item in rows])),
        "ul_mean_p1_p4": pcc.get("ul_mean_p1_p4"),
        "ec_mean_p1_p4": pcc.get("ec_mean_p1_p4"),
        "eligible": eligible,
        "ineligibility_reasons": [
            reason for condition, reason in (
                (observed != EXPECTED_EVALS, "missing_dev_phase"),
                (len(convergence) != 4 or not all(convergence), "lr_not_converged"),
                (endpoint is None, "missing_cq_dev_endpoint"),
            ) if condition
        ],
    }


def select(config_paths: list[Path]) -> dict[str, Any]:
    if not 1 <= len(config_paths) <= 8:
        raise ValueError("Review cho phép từ một đến tối đa tám cấu hình LR")
    rows = [_collect(path.resolve()) for path in config_paths]
    eligible = [item for item in rows if item["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item["ul_mean_p1_p4"]),
            -float(item["macro_f1_mean"]),
            float("inf") if item["ec_mean_p1_p4"] is None else float(item["ec_mean_p1_p4"]),
            float(item["fit_seconds_shared"]),
        ),
    )
    winner = ranked[0] if ranked else None
    return {
        "schema_version": 1,
        "status": "SELECTED" if winner else "NO_ELIGIBLE_CONFIG",
        "split": "dev_only",
        "test_read": False,
        "selection_rule": [
            "eligible iff four dev phases + converged fits + CQ endpoint",
            "maximize UL mean P1-P4",
            "maximize Macro-F1 mean",
            "minimize EC mean P1-P4",
            "minimize shared fit seconds",
        ],
        "evaluated_config_count": len(rows),
        "maximum_allowed_configs": 8,
        "configs": rows,
        "selected": None if winner is None else {
            "experiment_id": winner["experiment_id"],
            "config_path": winner["config_path"],
            "class_weight_policy": winner["class_weight_policy"],
            "learner_params": winner["learner_params"],
            "ul_mean_p1_p4": winner["ul_mean_p1_p4"],
            "ec_mean_p1_p4": winner["ec_mean_p1_p4"],
            "macro_f1_mean": winner["macro_f1_mean"],
            "mcc_mean": winner["mcc_mean"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--output", default="artifacts/cq_v1_m0_lr_tuning/lr_tuning_decision.json",
    )
    args = parser.parse_args()
    result = select([Path(item) for item in args.config])
    output = Path(args.output).resolve()
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
