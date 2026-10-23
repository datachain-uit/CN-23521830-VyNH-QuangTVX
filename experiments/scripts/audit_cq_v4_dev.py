"""Audit CQ V4 train/dev and optional capped diagnostic probes, without test data.

Exports only aggregate evidence. A completed probe is not proof of absence of
leakage. Raw masks require pre-imputation train/dev and are never reconstructed
from imputed zeros. Event-level cutoff validation remains an explicit pending
item when the runtime bundle cannot independently prove it.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.data.adapters.course_quality_v4 import _verify_sha256s
from skq_exp.training.metrics import compute_overall_metrics, compute_per_label_metrics
from scripts.run_cq_v1_m0_probes import _categorical_probe


def labels(frame):
    return frame.y_alpha_1.map({"E": 0, "G": 1, "A": 2}).to_numpy(int)


def distribution(frame):
    sizes = frame.groupby("course_id").size()
    shares = sizes / sizes.sum()
    return {"rows": len(frame), "courses": len(sizes), "families": int(frame.course_family_id.nunique()), "class_counts": {k: int(v) for k,v in frame.y_alpha_1.value_counts().items()}, "eligible_courses_n30": int((sizes >= 30).sum()), "largest_course_share": float(shares.max()), "effective_course_count": float(1 / (shares**2).sum())}


def audit(source: Path):
    _verify_sha256s(source)
    columns = json.loads((source / "learner_feature_columns.json").read_text())["columns"]
    selector = json.loads((source / "selector_feature_columns.json").read_text())["columns"]
    train = pd.read_parquet(source / "pool_b/train.parquet")
    full = pd.read_parquet(source / "fulltrain_a/train.parquet")
    dev = {f"P{i}": pd.read_parquet(source / f"pool_b/dev_phase{i}.parquet") for i in range(1,5)}
    all_frames = {"train": train, "fulltrain_a": full, **dev}
    checks = []
    def add(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    add("bundle_integrity", True, "SHA256SUMS.txt verified")
    add("no_test_data_in_runtime", not any(p.is_file() and "test" in p.name.lower() for p in source.rglob("*")), "Only runtime dev bundle inspected")
    forbidden = {"course_id", "course_family_id", "user_id", "phase", "split", "q_alpha_1", "y_alpha_1", "q_alpha_0p1", "y_alpha_0p1"}
    add("feature_allowlist", set(selector) <= set(columns) and not forbidden.intersection(columns), {"selector": len(selector), "learner": len(columns)})
    add("finite_features", all(np.isfinite(f[columns].to_numpy(float)).all() for f in all_frames.values()), "Ready numeric matrices; this is not an independent scaler-fit audit")
    add("unique_row_and_enrollment_phase", all(not f.row_id.duplicated().any() and not f.duplicated(["user_id","course_id","phase"]).any() for f in all_frames.values()), "Phase distinguishes repeated enrollment snapshots")
    add("train_dev_family_disjoint", all(not set(train.course_family_id).intersection(f.course_family_id) for f in dev.values()), "Family-group split")
    add("label_thresholds", all(np.array_equal(np.where(f.q_alpha_1 < .2,"E", np.where(f.q_alpha_1 < .4,"G","A")), f.y_alpha_1.astype(str)) and f.q_alpha_1.between(0,1).all() for f in all_frames.values()), "Checks tau(q), not the upstream component formula")
    pairs = lambda f: set(map(tuple, f[["user_id","course_id"]].to_numpy()))
    add("fulltrain_a_cohort", pairs(train) == pairs(full) and not full.duplicated(["user_id","course_id"]).any(), "lambda=1 train snapshot has same cohort")
    add("four_phase_cohorts", all(pairs(train[train.phase==p]) == pairs(full) for p in dev), "Same training enrollments in P1-P4")
    add("dev_phase_cohort", all(pairs(f) == pairs(dev["P1"]) for f in dev.values()), "Same dev enrollments in P1-P4")
    cutoff_ok = True
    for key, f in all_frames.items():
        start = pd.to_datetime(f.enroll_time)
        end = pd.to_datetime(f.course_end)
        cutoff = pd.to_datetime(f.cutoff_time)
        if key == "fulltrain_a":
            cutoff_ok &= bool((abs((cutoff-end).dt.total_seconds()) <= 1).all())
        else:
            frac = f.phase.map({"P1":.25,"P2":.5,"P3":.75,"P4":.9})
            expected = start + (end-start)*frac
            cutoff_ok &= bool((abs((cutoff-expected).dt.total_seconds()) <= 1).all())
    add("snapshot_cutoff_formula", cutoff_ok, "Cutoff metadata validated; feature event provenance requires source-event checks")
    prep = json.loads((source / "preprocessing.json").read_text())
    add("preprocessing_declaration", prep.get("fit_split") == "train", {"fit_split": prep.get("fit_split"), "fit_rows": prep.get("fit_rows"), "independent_refit_verified": False})
    return {"schema_version":1, "dataset":"CQ V4", "test_data_read":False, "checks":checks, "all_executed_checks_pass":all(c["status"]=="PASS" for c in checks), "full_review_checks_1_12_closed":False, "remaining_checks":["Full event-to-feature cutoff reconciliation and human review of 20 cases", "Upstream label-component formula/deviations supervisor approval", "Independent train-only preprocessing reconstruction", "Selector QP/exact-budget checks on the future dev panel"], "distribution":{k:distribution(f) for k,f in all_frames.items()}, "hashes":{name:sha256_file(source/name) for name in ("splits.json","preprocessing.json","selector_feature_columns.json","learner_feature_columns.json")}}, train, dev, columns


def probes(train, dev, columns, *, cap, raw_root=None):
    y = labels(train)
    ids = np.arange(len(train))
    selected, _ = train_test_split(ids, train_size=min(cap,len(ids)-1), stratify=y, random_state=20261022)
    result = {"scope":"capped_development_diagnostic", "train_fit_rows":len(selected), "train_rows_total":len(train), "test_data_read":False, "majority":{}}
    for phase, frame in dev.items():
        prediction = np.full(len(frame), np.bincount(y).argmax())
        result["majority"][phase] = compute_overall_metrics(labels(frame), prediction, labels=np.arange(3))[0]
    with threadpool_limits(limits=4):
        result["id_only_lr"] = {
            "train_fit_rows": len(selected),
            "course_id": _categorical_probe(train.iloc[selected], dev["P4"], ["course_id"], "y_alpha_1"),
            "teacher_school": _categorical_probe(train.iloc[selected], dev["P4"], ["teacher_set_code", "school_set_code"], "y_alpha_1"),
            "scope": "capped P4 diagnostic, not a production learner",
        }
        scaler = StandardScaler().fit(train.iloc[selected][columns])
        shuffled = np.random.default_rng(20261022).permutation(y[selected])
        model = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(scaler.transform(train.iloc[selected][columns]), shuffled)
        result["shuffle_labels_lr"] = {"fit_converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught), "learner":"LR diagnostic; does not replace required XGBoost shuffled-label probe", "per_phase":{phase:compute_overall_metrics(labels(f), model.predict(scaler.transform(f[columns])), model.predict_proba(scaler.transform(f[columns])), np.arange(3))[0] for phase,f in dev.items()}}
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.05, n_jobs=4, device="cpu", tree_method="hist", random_state=42, eval_metric="mlogloss")
        xgb.fit(train.iloc[selected][columns].to_numpy(np.float32), shuffled)
        result["shuffle_labels_xgb"] = {"learner":"XGB CPU; capped diagnostic; no early stopping", "train_fit_rows":len(selected), "n_estimators":100, "per_phase":{phase:compute_overall_metrics(labels(f), xgb.predict(f[columns].to_numpy(np.float32)), xgb.predict_proba(f[columns].to_numpy(np.float32)), np.arange(3))[0] for phase,f in dev.items()}}
        if raw_root:
            raw = pd.concat([pd.read_parquet(raw_root / "supervised_raw" / p / "train.parquet", columns=["row_id","y_alpha_1",*columns]) for p in dev], ignore_index=True)
            ordered = raw.set_index("row_id").loc[train.row_id]
            if not np.array_equal(ordered.y_alpha_1.to_numpy(), train.y_alpha_1.to_numpy()):
                raise ValueError("Raw/ready labels not aligned")
            mask = ordered[columns].isna().to_numpy(np.float32)
            model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(mask[selected], y[selected])
            phases = {}
            for phase, frame in dev.items():
                raw_dev = pd.read_parquet(raw_root / "supervised_raw" / phase / "dev.parquet", columns=["row_id",*columns]).set_index("row_id").loc[frame.row_id]
                x = raw_dev[columns].isna().to_numpy(np.float32)
                phases[phase] = compute_overall_metrics(labels(frame), model.predict(x), model.predict_proba(x), np.arange(3))[0]
            result["missing_mask_lr"] = {"fit_converged":not any(issubclass(w.category,ConvergenceWarning) for w in caught), "per_phase":phases,"production_features_changed":False}
        else:
            result["missing_mask_lr"] = {"status":"PENDING_RAW_TRAIN_DEV"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--raw-root")
    parser.add_argument("--probe-cap", type=int, default=20000)
    args = parser.parse_args()
    payload, train, dev, columns = audit(Path(args.input_dir))
    if args.probes:
        payload["probes"] = probes(train,dev,columns,cap=args.probe_cap,raw_root=Path(args.raw_root) if args.raw_root else None)
    atomic_json(Path(args.output),payload)
    print(json.dumps({"output":args.output,"executed_checks":len(payload["checks"]),"pass":payload["all_executed_checks_pass"]}))


if __name__ == "__main__":
    main()
