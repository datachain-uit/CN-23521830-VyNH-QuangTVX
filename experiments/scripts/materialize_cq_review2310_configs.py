"""Materialize registered next-stage configs AFTER a measured learner choice.

This never chooses a winner, runs a fit or grants approval. The caller supplies
the selected config and actual selection evidence. Fresh output directories
preserve the original configs and give a provenance receipt for each stage.
No panel, test or M1 freeze configuration is generated here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.config import ExperimentConfig


def configs_for_stage(base: ExperimentConfig, stage: str) -> dict[str, ExperimentConfig]:
    base.validate()
    if (base.feature_view != "cd3" or base.learner_ids != ("xgb",)
            or base.method_ids != ("c00_full_train",) or base.training_phases
            or base.label_shuffle_seed is not None or base.class_weight_policy != "natural"
            or base.model_seed != 42 or not base.test_locked
            or (base.early_stopping or {}).get("patience") != 50):
        raise ValueError("Need selected natural CD3 FullTrain-B XGB seed42, patience50")
    params = dict(base.learner_params["xgb"])
    outputs = {}

    def add(name, **changes):
        item = replace(base, experiment_id=f"cq_v4_r2310_{name}", **changes)
        item.validate()
        outputs[name] = item

    if stage == "grid":
        for depth in (4, 6, 8):
            for rate in (0.05, 0.1):
                add(f"grid_d{depth}_lr{rate:g}", learner_params={"xgb": {
                    **params, "max_depth": depth, "learning_rate": rate}},
                    protocol_id="cq_v4_review2310_xgb_grid6_v1")
    elif stage == "min-child":
        if params.get("max_depth") not in (4, 6, 8) or params.get("learning_rate") not in (.05, .1):
            raise ValueError("min-child stage needs a winner from the registered six-config grid")
        for child in (1, 10):
            add(f"min_child_{child}", learner_params={"xgb": {**params, "min_child_weight": child}},
                protocol_id="cq_v4_review2310_xgb_child2_v1")
    elif stage == "followups":
        policy = dict(base.artifact_policy or {})
        policy.pop("diagnostic_rounds", None)
        base = replace(base, artifact_policy=policy)
        for seed in (42, 43, 44):
            add(f"fullB_ms{seed}", model_seed=seed, budget_ratio=1.0)
        add("fullA_ms42", method_ids=("c00_full_train_a",), budget_ratio=1.0)
        for ratio in (.005, .01, .02, .05):
            add(f"c02_rho{ratio:g}", method_ids=("c02_stratified_random",), budget_ratio=ratio,
                selector_seeds=(11, 29, 47, 71, 101))
        for seed in (20261022, 20261023, 20261024):
            add(f"d04_shuffle{seed}", label_shuffle_seed=seed,
                evidence_role="diagnostic_shuffled_labels", budget_ratio=1.0)
    else:
        raise ValueError("Unknown registered materialization stage")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--stage", choices=("grid", "min-child", "followups"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.selection_evidence.is_file():
        raise FileNotFoundError("Measured selection evidence is required; not a planned result")
    configs = configs_for_stage(ExperimentConfig.from_json(args.selected_config), args.stage)
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory; no registered config is overwritten")
    args.output_dir.mkdir(parents=True)
    for name, config in configs.items():
        atomic_json(args.output_dir / f"{name}.json", config.to_dict())
    atomic_json(args.output_dir / "materialization_receipt.json", {
        "stage": args.stage, "source_config_sha256": sha256_file(args.selected_config),
        "source_selection_evidence_sha256": sha256_file(args.selection_evidence),
        "files": list(configs), "execution_status": "not_run", "teacher_approval": "not_granted_by_this_script",
        "panel_or_test_configs_generated": False,
    })
    print(f"Prepared {len(configs)} configs; no fit or test performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
