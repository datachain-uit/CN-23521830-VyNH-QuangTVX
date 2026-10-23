"""Khóa rho* M0 từ đúng FullTrain-B và C02 dev theo QĐ12.

Script chỉ đọc run ledger development, yêu cầu đủ P1--P4 và seed 11/29/47 cho
bốn mức 0.5/1/2/5%. Mọi run lỗi hoặc trùng đều chặn decision; test không được đọc.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.config import ExperimentConfig
from skq_exp.experiments.headroom import (
    HEADROOM_RATIOS,
    HEADROOM_SEEDS,
    choose_headroom_budget,
)


DEV_PHASES = tuple(f"dev_phase{index}" for index in range(1, 5))


def _read_ledger(artifact_root: Path, experiment_id: str) -> list[dict[str, Any]]:
    path = artifact_root / experiment_id / "run_ledger.json"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu run ledger: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Run ledger không phải list: {path}")
    return payload


def _seed_ul(
    rows: list[dict[str, Any]],
    *,
    method_id: str,
    learner_id: str,
    expected_seeds: tuple[int, ...],
) -> dict[int, float]:
    """Lấy mean UL P1--P4; không coi phase là replicate độc lập."""
    result = {}
    for seed in expected_seeds:
        selected = [
            row for row in rows
            if row.get("method_id") == method_id
            and row.get("learner_id") == learner_id
            and int(row.get("selector_seed", -1)) == seed
        ]
        if any(row.get("status") != "success" for row in selected):
            raise ValueError(f"Có run không success cho {method_id}/seed={seed}")
        by_phase = {}
        for row in selected:
            phase = row.get("evaluation_split")
            if phase in by_phase:
                raise ValueError(f"Trùng phase {phase} cho {method_id}/seed={seed}")
            by_phase[phase] = float(row["macro_f1"])
        if set(by_phase) != set(DEV_PHASES):
            raise ValueError(
                f"Thiếu/thừa dev phase cho {method_id}/seed={seed}: {sorted(by_phase)}"
            )
        values = np.asarray([by_phase[phase] for phase in DEV_PHASES], dtype=float)
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"UL không hợp lệ cho {method_id}/seed={seed}")
        result[seed] = float(values.mean())
    return result


def compute(
    artifact_root: Path,
    fulltrain_config: Path,
    c02_configs: list[Path],
    output: Path,
    *,
    expected_seeds: tuple[int, ...] = HEADROOM_SEEDS,
    protocol_id: str = "cq_v1_m0_headroom_qd12_v1",
) -> dict[str, Any]:
    fulltrain = ExperimentConfig.from_json(fulltrain_config)
    if fulltrain.method_ids != ("c00_full_train",) or "xgb" not in fulltrain.learner_ids:
        raise ValueError("FullTrain config phải chứa duy nhất C00 và learner XGB")
    fulltrain_values = _seed_ul(
        _read_ledger(artifact_root, fulltrain.experiment_id),
        method_id="c00_full_train", learner_id="xgb", expected_seeds=(0,),
    )

    c02_dev_ul = {}
    config_evidence = []
    policies = {fulltrain.class_weight_policy}
    for path in c02_configs:
        config = ExperimentConfig.from_json(path)
        if config.method_ids != ("c02_stratified_random",):
            raise ValueError(f"Headroom chỉ nhận C02: {path}")
        if config.learner_ids != ("xgb",) or config.selector_seeds != expected_seeds:
            raise ValueError(f"C02 phải dùng XGB và seed {expected_seeds}: {path}")
        if config.budget_ratio in c02_dev_ul:
            raise ValueError(f"Trùng rho={config.budget_ratio}")
        policies.add(config.class_weight_policy)
        c02_dev_ul[config.budget_ratio] = _seed_ul(
            _read_ledger(artifact_root, config.experiment_id),
            method_id="c02_stratified_random", learner_id="xgb",
            expected_seeds=expected_seeds,
        )
        config_evidence.append({
            "experiment_id": config.experiment_id,
            "budget_ratio": config.budget_ratio,
            "config_path": str(path),
            "config_sha256": sha256_file(path),
        })
    if set(c02_dev_ul) != set(HEADROOM_RATIOS):
        raise ValueError(f"Cần đúng bốn rho {HEADROOM_RATIOS}")
    if len(policies) != 1:
        raise ValueError(f"Class-weight policy không nhất quán: {sorted(policies)}")

    decision = choose_headroom_budget(
        full_train_b_ul=fulltrain_values[0], c02_dev_ul=c02_dev_ul,
        required_seeds=expected_seeds,
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        commit = None
    payload = {
        **decision,
        "protocol_id": protocol_id,
        "test_data_read": False,
        "phase_aggregation": "mean_P1_P4_within_seed",
        "replicate_unit": "selector_seed",
        "learner_id": "xgb",
        "class_weight_policy": next(iter(policies)),
        "repository_commit": commit,
        "fulltrain_config": {
            "experiment_id": fulltrain.experiment_id,
            "config_path": str(fulltrain_config),
            "config_sha256": sha256_file(fulltrain_config),
        },
        "c02_configs": sorted(config_evidence, key=lambda item: item["budget_ratio"]),
    }
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--fulltrain-config",
        default="configs/cq_v1_m0_fulltrain_b_dev_natural_xgb_v1.json",
    )
    parser.add_argument("--c02-config", action="append", required=True)
    parser.add_argument(
        "--output", default="artifacts/cq_v1_m0_headroom/headroom_decision.json",
    )
    args = parser.parse_args()
    payload = compute(
        Path(args.artifact_root).resolve(),
        Path(args.fulltrain_config).resolve(),
        [Path(path).resolve() for path in args.c02_config],
        Path(args.output).resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
