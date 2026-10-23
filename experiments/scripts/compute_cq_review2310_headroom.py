"""Apply the predeclared QD25/QD26 rule to COMPLETE dev log-loss inputs.

Input is an audited JSON with full_by_model_seed (42/43/44), c02_by_ratio
(four budgets, five seeds each) and prior_ul. Family CIs are produced from
aligned predictions separately; this scalar decision file never fabricates CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.experiments.cq_decisions_1023 import headroom_proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("split") != "dev" or payload.get("primary_endpoint") != "mean_log_loss_P1_P4":
        raise ValueError("Input must explicitly identify dev log-loss endpoint, not legacy F1")
    if not payload.get("learner_selection_reference") or not payload.get("feature_contract_hash"):
        raise ValueError("Need selected learner and feature contract provenance")
    result = headroom_proposal(
        {int(k): v for k, v in payload["full_by_model_seed"].items()},
        {float(r): {int(s): v for s, v in values.items()} for r, values in payload["c02_by_ratio"].items()},
        payload["prior_ul"],
    )
    result.update(source_sha256=sha256_file(args.input),
                  learner_selection_reference=payload["learner_selection_reference"],
                  feature_contract_hash=payload["feature_contract_hash"],
                  family_ci_status="requires_separate_aligned_prediction_bootstrap")
    if args.output.exists():
        raise FileExistsError("Use a fresh output; do not overwrite a prior proposal")
    atomic_json(args.output, result)
    print(json.dumps({"rho_star_proposed": result["rho_star"], "approval_status": result["approval_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
