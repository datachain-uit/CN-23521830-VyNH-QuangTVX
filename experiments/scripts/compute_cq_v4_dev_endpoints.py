"""CQ V4 dev endpoints: align row IDs, keep course/family roles, export aggregates.

Reads an existing artifact directory or tar archive. Verifies all four dev
splits against the immutable runtime bundle; no test predictions are accepted.
EC is evaluated per offering (course_id); family IDs define split/bootstrap
units, not a replacement for the course endpoint. A8 uses train-only priors.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from skq_exp.artifacts import atomic_json
from skq_exp.data.adapters.course_quality_v4 import _group_ids, _row_ids
from skq_exp.training.course_metrics import (
    class_conditional_means, class_prevalence, cq_endpoints,
    prior_correct_predictions,
)
from skq_exp.training.metrics import compute_overall_metrics, compute_per_label_metrics

LABELS = np.asarray(["E", "G", "A"])


def align_predictions(frame: pd.DataFrame, prediction: dict) -> dict:
    """Reject missing/duplicated/foreign IDs before matching labels and groups."""
    wanted = _row_ids(frame)
    observed = np.asarray(prediction["row_id"])
    if len(observed) != len(wanted) or len(np.unique(observed)) != len(observed):
        raise ValueError("Prediction row IDs have missing/duplicate rows")
    positions = pd.Index(observed).get_indexer(wanted)
    if np.any(positions < 0):
        raise ValueError("Prediction row IDs do not match dev source")
    aligned = {key: np.asarray(value)[positions] for key, value in prediction.items()}
    numeric = frame.y_alpha_1.map({"E": 0, "G": 1, "A": 2}).to_numpy(int)
    if not np.array_equal(aligned["y_true"], numeric):
        raise ValueError("Prediction labels do not match dev source")
    for name in ("course_id", "course_family_id"):
        if not np.array_equal(aligned[name], _group_ids(frame[name])):
            raise ValueError(f"Prediction {name} does not match source")
    return aligned


def compute(source: Path, artifacts: Path, *, n_min: int = 30, experiment_id: str | None = None) -> dict:
    train = pd.read_parquet(source / "pool_b/train.parquet")
    unique = train.drop_duplicates(["user_id", "course_id"])
    mu = class_conditional_means(unique.rename(columns={"q_alpha_1": "q", "y_alpha_1": "y"}))
    prior = class_prevalence(train.y_alpha_1)
    targets = {p: class_prevalence(train.loc[train.phase == p, "y_alpha_1"]) for p in ("P1", "P2", "P3", "P4")}
    dev = {f"dev_phase{i}": pd.read_parquet(source / f"pool_b/dev_phase{i}.parquet") for i in range(1, 5)}
    archive = tarfile.open(artifacts) if artifacts.is_file() else None
    if archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]
        read = lambda name: archive.extractfile(name).read()
    else:
        names = [p.relative_to(artifacts).as_posix() for p in artifacts.rglob("manifest.json")]
        read = lambda name: (artifacts / name).read_bytes()
    grouped = {}
    try:
        for name in names:
            if not name.endswith("/manifest.json") or "/experiments/" not in "/" + name:
                continue
            manifest = json.loads(read(name))
            evaluation = manifest.get("evaluation_split", "")
            if evaluation not in dev:
                continue
            if manifest.get("dataset_id") != "course_quality_v4":
                continue
            if experiment_id is not None and manifest.get("experiment_id") != experiment_id:
                continue
            stem = name.removesuffix("manifest.json")
            key = tuple(manifest[k] for k in ("experiment_id", "method_id", "selector_seed", "learner_id", "model_seed"))
            if evaluation in grouped.setdefault(key, {}):
                raise ValueError(f"Duplicate evaluation identity: {key}/{evaluation}")
            with np.load(io.BytesIO(read(stem + "predictions.npz")), allow_pickle=False) as p:
                prediction = align_predictions(dev[evaluation], dict(p))
            grouped[key][evaluation] = (prediction, json.loads(read(stem + "cost.json")))
        runs = []
        for key, items in sorted(grouped.items()):
            if set(items) != set(dev):
                raise ValueError(f"Missing dev phase for {key}")
            frames, overall, per_label, confusion = [], {}, {}, {}
            source_priors = []
            for evaluation, (pred, cost) in sorted(items.items()):
                raw = dev[evaluation]
                phase = "P" + evaluation[-1]
                f = pd.DataFrame({
                    "run_id": key[0], "course_id": raw.course_id.to_numpy(),
                    "course_root_id": raw.course_family_id.to_numpy(),
                    "learner_id": raw.user_id.to_numpy(), "phase": phase,
                    "q_true": raw.q_alpha_1.to_numpy(), "y_true": raw.y_alpha_1.to_numpy(),
                    "y_pred": LABELS[pred["y_pred"].astype(int)],
                    **{f"p_{label}": pred["y_prob"][:, i].astype(np.float64) for i, label in enumerate(LABELS)},
                })
                frames.append(f)
                overall[phase], _ = compute_overall_metrics(pred["y_true"], pred["y_pred"], pred["y_prob"], labels=np.arange(3))
                counts = raw.groupby("course_id").course_id.transform("size").to_numpy()
                overall[phase]["ul_course_equal"] = float(f1_score(pred["y_true"], pred["y_pred"], labels=[0,1,2], average="macro", sample_weight=1.0/counts, zero_division=0))
                per_label[phase] = compute_per_label_metrics(pred["y_true"], pred["y_pred"], pred["y_prob"], np.arange(3))
                matrix = np.zeros((3,3), dtype=int)
                np.add.at(matrix, (pred["y_true"].astype(int), pred["y_pred"].astype(int)), 1)
                confusion[phase] = matrix.tolist()
                effective = cost["effective_class_prior"]
                source_priors.append({LABELS[int(label)]: float(prob) for label, prob in zip(effective["labels"], effective["probabilities"], strict=True)})
            if any(not np.allclose([p[l] for l in LABELS], [source_priors[0][l] for l in LABELS]) for p in source_priors):
                raise ValueError("Effective prior differs across phases of one fitted model")
            private = pd.concat(frames, ignore_index=True)
            pcc = cq_endpoints(private, mu, n_min=n_min, mode="pcc")
            corrected = prior_correct_predictions(private, source_priors[0], targets)
            runs.append({
                "status": "success", "experiment_id": key[0], "method_id": key[1],
                "selector_seed": key[2], "learner_id": key[3], "model_seed": key[4],
                "n_min": n_min, "mu_train": mu, "root_policy": "course_family_id",
                "course_endpoint_unit": "course_id", "bootstrap_unit": "course_family_id",
                "pcc": pcc, "cc_sensitivity": cq_endpoints(private, mu, n_min=n_min, mode="cc"),
                "a8_prior_correction": {"retrained": False, "assumption": "label_shift_only", "source_prior": source_priors[0], "target_prior_by_phase": targets, "pcc": cq_endpoints(corrected, mu, n_min=n_min)},
                "overall_by_phase": overall, "per_label_by_phase": per_label,
                "confusion_matrices": confusion,
                "mean_metrics": {k: float(np.mean([v[k] for v in overall.values()])) for k in overall["P1"] if all(v.get(k) is not None for v in overall.values())},
            })
    finally:
        if archive:
            archive.close()
    if not runs:
        raise ValueError("No CQ V4 dev predictions found")
    return {"schema_version": 1, "split": "dev_only", "test_data_read": False, "row_level_artifacts_saved": False, "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    result = compute(Path(args.input_dir), Path(args.artifacts), n_min=args.n_min, experiment_id=args.experiment_id)
    atomic_json(Path(args.output), result)
    print(json.dumps({"runs": len(result["runs"]), "output": args.output}))


if __name__ == "__main__":
    main()
