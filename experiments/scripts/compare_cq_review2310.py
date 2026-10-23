"""Paired D13 log-loss CI and top-course contributions from private dev predictions.

Accepts an unpacked artifact directory, aligns both runs to the immutable dev
source and never reads test. Four-phase CIs resample families jointly 2000
times. Diagnostic S is intentionally not treated as a four-phase candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .compute_cq_v4_dev_endpoints import align_predictions
else:
    from compute_cq_v4_dev_endpoints import align_predictions
from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.data.cq_patch_1023 import course_contributions, row_log_loss
from skq_exp.experiments.cq_decisions_1023 import paired_family_logloss_ci


def _read_run(artifacts: Path, experiment: str, dev: dict):
    parts, provenance = {}, {}
    for path in artifacts.rglob("manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        name = manifest.get("evaluation_split")
        if manifest.get("experiment_id") != experiment or name not in dev:
            continue
        if manifest.get("dataset_id") != "course_quality_v4":
            raise ValueError("Comparison requires CQ V4")
        phase = "P" + name[-1]
        if phase in parts:
            raise ValueError("Comparison requires exactly one fitted run per experiment, not mixed seeds/models")
        prediction_file = path.parent / "predictions.npz"
        with np.load(prediction_file, allow_pickle=False) as payload:
            aligned = align_predictions(dev[name], dict(payload))
        p = np.asarray(aligned["y_prob"], float)
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Invalid probabilities")
        parts[phase] = p / p.sum(axis=1, keepdims=True)
        provenance[phase] = {"manifest_sha256": sha256_file(path), "prediction_sha256": sha256_file(prediction_file)}
    if set(parts) != {"P1", "P2", "P3", "P4"}:
        raise ValueError(f"Need complete four-phase fitted run for {experiment}; S is P4-only")
    return parts, provenance


def compare(source: Path, artifacts: Path, experiment_a: str, experiment_b: str, *, n_bootstrap=2000):
    dev = {f"dev_phase{i}": pd.read_parquet(source / f"pool_b/dev_phase{i}.parquet") for i in range(1, 5)}
    a, provenance_a = _read_run(artifacts, experiment_a, dev)
    b, provenance_b = _read_run(artifacts, experiment_b, dev)
    labels = {"P"+name[-1]: f.y_alpha_1.map({"E": 0, "G": 1, "A": 2}).to_numpy(int) for name, f in dev.items()}
    families = {"P"+name[-1]: f.course_family_id.to_numpy() for name, f in dev.items()}
    paired = paired_family_logloss_ci(a, b, labels, families, n_bootstrap=n_bootstrap)
    loss_a, loss_b = {p: row_log_loss(a[p], labels[p]) for p in a}, {p: row_log_loss(b[p], labels[p]) for p in b}
    courses = {"P"+name[-1]: f.course_id.to_numpy() for name, f in dev.items()}
    contributions = {p: course_contributions(loss_a[p], loss_b[p], courses[p]) for p in a}
    phases = ("P1", "P2", "P3", "P4")
    aggregate = course_contributions(np.concatenate([loss_a[p] for p in phases]),
                                     np.concatenate([loss_b[p] for p in phases]),
                                     np.concatenate([courses[p] for p in phases]))
    return {"comparison": f"{experiment_a} minus {experiment_b}", "split": "dev", "test_data_read": False,
            "primary_endpoint": "mean_log_loss_P1_P4", "paired_family_ci": paired,
            "course_contributions_by_phase": contributions, "course_contributions_mean_phases": aggregate,
            "source_predictions": {"A": provenance_a, "B": provenance_b},
            "teacher_approval": "not_granted_by_this_comparison"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--a-experiment", required=True)
    parser.add_argument("--b-experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh comparison report")
    atomic_json(args.output, compare(args.input_dir, args.artifacts, args.a_experiment, args.b_experiment))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
