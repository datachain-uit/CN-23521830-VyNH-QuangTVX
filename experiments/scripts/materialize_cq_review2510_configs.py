"""Prepare QD29 final A/B/S and Y29 qa23_all ONLY after measured learner selection.

Reuse the registered Review23 follow-ups without modifying historical configs.
No model fit, panel approval, test configuration or freeze is produced here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.config import ExperimentConfig
if __package__:
    from .materialize_cq_review2310_configs import configs_for_stage
else:
    from materialize_cq_review2310_configs import configs_for_stage


def followups(base, qa23_processed_root: Path):
    if base.feature_variant is not None:
        raise ValueError("Selected learner must come from main CD3, never qa23_all")
    if qa23_processed_root.resolve() == base.paths.processed_root.resolve():
        raise ValueError("qa23_all requires a separate processed_root")
    configs = configs_for_stage(base, "followups")
    configs = {n: replace(c, experiment_id=c.experiment_id.replace("r2310", "r2510"))
               for n, c in configs.items()}
    configs["S_final_ms42"] = replace(configs["fullB_ms42"],
        experiment_id="cq_v4_r2510_S_final_ms42", training_phases=("P4",))
    configs["qa23_all_ms42"] = replace(configs["fullB_ms42"],
        experiment_id="cq_v4_r2510_qa23_all_ms42", feature_variant="qa23_all",
        evidence_role="diagnostic_qa23_all_not_candidate",
        paths=replace(base.paths, processed_root=qa23_processed_root.resolve()))
    for c in configs.values():
        c.validate()
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--stage", choices=("grid", "min-child", "followups"), default="followups")
    parser.add_argument("--qa23-processed-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.selection_evidence.read_text(encoding="utf-8"))
    if (evidence.get("split") != "dev" or evidence.get("status") != "selected"
            or evidence.get("selected_config_sha256") != sha256_file(args.selected_config)
            or not evidence.get("feature_contract_hash") or evidence.get("test_data_read") is not False
            or evidence.get("d13_step1_gate_passed") is not True):
        raise ValueError("Need measured CD3 selection evidence and passed D13 step1 gate")
    base = ExperimentConfig.from_json(args.selected_config)
    if args.stage == "followups":
        if args.qa23_processed_root is None:
            raise ValueError("followups needs separate --qa23-processed-root")
        configs = followups(base, args.qa23_processed_root)
    else:
        if base.feature_variant is not None:
            raise ValueError("Grid cannot select qa23_all")
        configs = configs_for_stage(base, args.stage)
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    for name, config in configs.items():
        atomic_json(args.output_dir / f"{name}.json", config.to_dict())
    atomic_json(args.output_dir / "materialization_receipt.json", {
        "review": "2026-10-25", "stage": args.stage, "execution_status": "not_run", "configs": list(configs),
        "selection_evidence_sha256": sha256_file(args.selection_evidence),
        "selected_config_sha256": sha256_file(args.selected_config),
        "gate_checked_at": "step1_only", "final_A_S_gate_rechecked": False,
        "qa23_all_is_candidate": False, "panel_or_test_configs_generated": False})
    print(f"Prepared {len(configs)} follow-ups; no fit, panel, or test")


if __name__ == "__main__":
    main()
