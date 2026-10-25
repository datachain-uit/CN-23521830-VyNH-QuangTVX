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
from sklearn.metrics import f1_score, log_loss

from skq_exp.artifacts import atomic_json
from skq_exp.data.adapters.course_quality_v4 import _group_ids, _row_ids
from skq_exp.training.course_metrics import (
    class_conditional_means, class_prevalence, cq_endpoints,
    prior_correct_predictions,
)
from skq_exp.training.metrics import compute_overall_metrics, compute_per_label_metrics
from skq_exp.data.cq_patch_1023 import (
    calibration_in_the_large, subgroup_log_loss, ul_log_loss,
)
from skq_exp.data.cq_patch_1025 import a8_defined
from skq_exp.experiments.cq_review_1025 import e_phase_report, DECISION_ENDPOINT

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


def compute(source: Path, artifacts: Path, *, n_min: int = 30, experiment_id: str | None = None,
            review_1023: bool = False, review_1025: bool = False, n_bootstrap: int = 2000) -> dict:
    review_1023 = review_1023 or review_1025
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
            calibration, subgroups, fit_diagnostics = {}, {}, {}
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
                if review_1023:
                    if cost.get("feature_view") not in {"cd3", "v4"}:
                        raise ValueError("Review 23/10 requires feature-view provenance, not legacy run relabeling")
                    overall[phase]["log_loss_course_equal"] = float(log_loss(
                        pred["y_true"], pred["y_prob"], labels=[0, 1, 2], sample_weight=1.0/counts,
                    ))
                    calibration[phase] = calibration_in_the_large(pred["y_prob"], pred["y_true"])
                    user_groups = np.where(raw.user_id.isin(train.user_id), "seen", "new")
                    subgroups[phase] = {"user": subgroup_log_loss(pred["y_prob"], pred["y_true"], user_groups)}
                    if "teacher_subgroup" in raw:
                        subgroups[phase]["teacher"] = subgroup_log_loss(
                            pred["y_prob"], pred["y_true"], raw.teacher_subgroup.fillna("missing").to_numpy(),
                        )
                    else:
                        subgroups[phase]["teacher"] = {"status": "missing_data_team_teacher_mapping"}
                    fit_diagnostics[phase] = cost
                per_label[phase] = compute_per_label_metrics(pred["y_true"], pred["y_pred"], pred["y_prob"], np.arange(3))
                matrix = np.zeros((3,3), dtype=int)
                np.add.at(matrix, (pred["y_true"].astype(int), pred["y_pred"].astype(int)), 1)
                confusion[phase] = matrix.tolist()
                effective = cost["effective_class_prior"]
                source_prior = {label: 0.0 for label in LABELS}
                for label, prob in zip(effective["labels"], effective["probabilities"], strict=True):
                    source_prior[LABELS[int(label)]] = float(prob)
                if (not np.isfinite(list(source_prior.values())).all()
                        or min(source_prior.values()) < 0
                        or not np.isclose(sum(source_prior.values()), 1)):
                    raise ValueError("Invalid effective source prior")
                source_priors.append(source_prior)
            if any(not np.allclose([p[l] for l in LABELS], [source_priors[0][l] for l in LABELS]) for p in source_priors):
                raise ValueError("Effective prior differs across phases of one fitted model")
            private = pd.concat(frames, ignore_index=True)
            pcc = cq_endpoints(private, mu, n_min=n_min, mode="pcc")
            a8_undefined = not a8_defined([source_priors[0][label] for label in LABELS])
            corrected = None if a8_undefined else prior_correct_predictions(private, source_priors[0], targets)
            runs.append({
                "status": "success", "experiment_id": key[0], "method_id": key[1],
                "selector_seed": key[2], "learner_id": key[3], "model_seed": key[4],
                "n_min": n_min, "mu_train": mu, "root_policy": "course_family_id",
                "course_endpoint_unit": "course_id", "bootstrap_unit": "course_family_id",
                "pcc": pcc, "cc_sensitivity": cq_endpoints(private, mu, n_min=n_min, mode="cc"),
                "a8_prior_correction": {"retrained": False, "assumption": "label_shift_only", "source_prior": source_priors[0], "target_prior_by_phase": targets,
                    "a8_undefined": a8_undefined,
                    "status": "A8 không xác định (thiếu lớp)" if a8_undefined else "defined",
                    "pcc": None if a8_undefined else cq_endpoints(corrected, mu, n_min=n_min)},
                "overall_by_phase": overall, "per_label_by_phase": per_label,
                "confusion_matrices": confusion,
                "mean_metrics": {k: float(np.mean([v[k] for v in overall.values()])) for k in overall["P1"] if all(v.get(k) is not None for v in overall.values())},
            })
            if review_1023:
                def ll(frame):
                    parts = {p: frame[frame.phase == p] for p in ("P1", "P2", "P3", "P4")}
                    return ul_log_loss(
                        # Restore row normalization after float32 probability storage.
                        {p: (lambda v: v / v.sum(axis=1, keepdims=True))(
                            f[["p_E", "p_G", "p_A"]].to_numpy(float)) for p, f in parts.items()},
                        {p: f.y_true.map({"E": 0, "G": 1, "A": 2}).to_numpy(int) for p, f in parts.items()},
                    )
                runs[-1].update({
                    "primary_endpoint": "mean_log_loss_P1_P4", "direction": "lower_is_better",
                    "ul_log_loss": ll(private),
                    "macro_f1_secondary": runs[-1]["mean_metrics"]["f1_macro"],
                    "calibration_in_the_large": calibration, "subgroup_log_loss": subgroups,
                    "fit_diagnostics_by_phase": fit_diagnostics,
                    "legacy_pcc_ul_fields_are_macro_f1_not_primary": True,
                })
                runs[-1]["a8_prior_correction"]["ul_log_loss"] = None if a8_undefined else ll(corrected)
            if review_1025:
                phase_priors = next(iter(items.values()))[1].get("effective_class_prior_by_training_phase")
                if phase_priors is None and key[1] in {"c00_full_train", "c00_full_train_a"}:
                    phase_priors = {ph: source_priors[0] for ph in targets}
                dp = {ph: (float(sum((phase_priors[ph][lab] - targets[ph][lab])*mu[lab] for lab in LABELS))
                           if phase_priors and phase_priors.get(ph) is not None else None) for ph in targets}
                def e_report(frame):
                    parts = {p: frame[frame.phase == p] for p in ("P1", "P2", "P3", "P4")}
                    return e_phase_report(
                        {p: f[["p_E", "p_G", "p_A"]].to_numpy(float) for p, f in parts.items()},
                        {p: f.y_true.map({"E": 0, "G": 1, "A": 2}).to_numpy(int) for p, f in parts.items()},
                        {p: f.course_root_id.to_numpy() for p, f in parts.items()}, n_bootstrap=n_bootstrap)
                raw_ul = runs[-1]["ul_log_loss"]["ul"]
                corrected_ul = None if a8_undefined else runs[-1]["a8_prior_correction"]["ul_log_loss"]["ul"]
                runs[-1].update(primary_endpoint=DECISION_ENDPOINT,
                    ul_raw=raw_ul, ul_a8=corrected_ul,
                    ul_decision=raw_ul if a8_undefined else corrected_ul,
                    a8_undefined=a8_undefined, decision_probability_basis="raw_missing_class_fallback" if a8_undefined else "A8_to_Pool_B_by_phase",
                    e_class_raw=e_report(private),
                    e_class_a8=None if a8_undefined else e_report(corrected),
                    dp_by_phase=dp,
                    dp_basis="effective_selected_prior_conditional_on_training_phase",
                    dp_missing_phase_policy="undefined_not_replaced_by_global_prior",
                    ec_decision_basis="raw_missing_class_fallback" if a8_undefined else "A8",
                    ec_decision=runs[-1]["pcc"] if a8_undefined else runs[-1]["a8_prior_correction"]["pcc"],
                    feature_variant=next(iter(items.values()))[1].get("feature_variant", "main"))
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
    parser.add_argument("--review-1023", action="store_true", help="Report QD25 primary LL and D13 diagnostics")
    parser.add_argument("--review-1025", action="store_true", help="QD30 ULA8 decisions and QD32 E metrics/cluster CI")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = compute(Path(args.input_dir), Path(args.artifacts), n_min=args.n_min,
                     experiment_id=args.experiment_id, review_1023=args.review_1023,
                     review_1025=args.review_1025, n_bootstrap=args.n_bootstrap)
    if Path(args.output).exists():
        raise FileExistsError("Use a fresh endpoint output; preserve historical results")
    atomic_json(Path(args.output), result)
    print(json.dumps({"runs": len(result["runs"]), "output": args.output}))


if __name__ == "__main__":
    main()
