"""Chạy D02--D06 M0 trên train/dev CQ v1 mà không mở test.

D03 chỉ dùng mask thiếu của ``supervised_raw`` làm probe chẩn đoán;
mask không được thêm vào learner/selector view. D04 dùng matrix đã được
fit một lần trên Pool B train. D06 sinh đúng 20 dòng private để kiểm
timestamp thủ công; JSON công khai chỉ giữ hash của cặp learner--course.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


PHASES = ("P1", "P2", "P3", "P4")
LABEL_TO_INT = {"E": 0, "G": 1, "A": 2}


def _labels(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].astype(str).map(LABEL_TO_INT).to_numpy(dtype=int)


def _majority(y_train: np.ndarray, y_dev: np.ndarray) -> dict:
    label = int(np.bincount(y_train, minlength=3).argmax())
    prediction = np.full(len(y_dev), label, dtype=int)
    return {"majority_label": label, "macro_f1": float(f1_score(y_dev, prediction, labels=[0, 1, 2], average="macro", zero_division=0))}


def _categorical_probe(train: pd.DataFrame, dev: pd.DataFrame, columns: list[str], target: str) -> dict:
    model = make_pipeline(
        ColumnTransformer([("id", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), columns)]),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=None),
    )
    y_train = _labels(train, target)
    y_dev = _labels(dev, target)
    model.fit(train[columns], y_train)
    prediction = model.predict(dev[columns])
    unseen = {
        column: float((~dev[column].astype(str).isin(set(train[column].astype(str)))).mean())
        for column in columns
    }
    return {
        "columns": columns,
        "macro_f1": float(f1_score(y_dev, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "unseen_rate_dev": unseen,
    }


def _stratified_cap(X: np.ndarray, y: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Lấy mẫu train có trần nhưng giữ đủ mọi lớp."""
    if len(y) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        size = max(1, int(round(cap * len(candidates) / len(y))))
        selected.append(rng.choice(candidates, size=min(size, len(candidates)), replace=False))
    indices = np.concatenate(selected)
    if len(indices) > cap:
        indices = rng.choice(indices, size=cap, replace=False)
    rng.shuffle(indices)
    return X[indices], y[indices]


def _missingness_probe(
    source: Path, target: str, numeric_columns: list[str], seed: int,
) -> dict:
    raw_root = source / "supervised_raw"
    if not all(
        (raw_root / phase / f"{split}.parquet").is_file()
        for phase in PHASES for split in ("train", "dev")
    ):
        return {"status": "BLOCKED_RAW_PREIMPUTATION_REQUIRED"}
    train_frames = [
        pd.read_parquet(raw_root / phase / "train.parquet", columns=[target, *numeric_columns])
        for phase in PHASES
    ]
    train = pd.concat(train_frames, ignore_index=True)
    X_train = train[numeric_columns].isna().to_numpy(dtype=np.float32)
    y_train = _labels(train, target)
    X_fit, y_fit = _stratified_cap(X_train, y_train, 200_000, seed)
    model = LogisticRegression(
        max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed,
    )
    model.fit(X_fit, y_fit)
    per_phase: dict[str, dict[str, float | int | dict[str, float]]] = {}
    for phase in PHASES:
        dev = pd.read_parquet(
            raw_root / phase / "dev.parquet", columns=[target, *numeric_columns],
        )
        X_dev = dev[numeric_columns].isna().to_numpy(dtype=np.float32)
        y_dev = _labels(dev, target)
        probability = model.predict_proba(X_dev)
        prediction = probability.argmax(axis=1)
        try:
            auc = float(roc_auc_score(y_dev, probability, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        per_phase[phase] = {
            "rows": int(len(dev)),
            "macro_f1": float(f1_score(y_dev, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "macro_ovr_roc_auc": auc,
            "missing_rate_by_column": {
                column: float(dev[column].isna().mean()) for column in numeric_columns
                if bool(dev[column].isna().any())
            },
        }
    return {
        "status": "PASS_PROBE_COMPLETED",
        "model": "balanced_multinomial_logistic_on_missing_mask_only",
        "train_rows_total": int(len(train)),
        "train_rows_fit": int(len(y_fit)),
        "missing_mask_used_as_production_feature": False,
        "per_phase": per_phase,
        "macro_ovr_roc_auc_mean": float(np.nanmean([row["macro_ovr_roc_auc"] for row in per_phase.values()])),
    }


def _pair_hash(user_id: object, course_id: object) -> str:
    return hashlib.sha256(f"{user_id}\t{course_id}".encode("utf-8")).hexdigest()


def _cutoff_probe(source: Path, private_output: Path, seed: int) -> dict:
    audit_root = source / "audit"
    if not all((audit_root / phase).exists() for phase in PHASES):
        return {"status": "BLOCKED_RAW_EVENT_TIMESTAMPS_REQUIRED"}
    rng = np.random.default_rng(seed)
    private_rows: list[pd.DataFrame] = []
    public_rows: list[dict[str, object]] = []
    for phase in PHASES:
        audit = pd.read_parquet(audit_root / phase)
        required = {
            "user_id", "course_id", "enroll_time", "cutoff_time",
            "min_feature_event_time", "max_feature_event_time",
            "n_out_of_window_events",
        }
        missing = sorted(required - set(audit.columns))
        if missing:
            return {"status": "BLOCKED_AUDIT_SCHEMA", "missing_columns": missing}
        if len(audit) < 5:
            return {"status": "BLOCKED_TOO_FEW_AUDIT_ROWS", "phase": phase}
        selected = audit.iloc[rng.choice(len(audit), size=5, replace=False)].copy()
        selected["phase"] = phase
        selected["max_not_after_cutoff"] = (
            pd.to_datetime(selected["max_feature_event_time"], errors="coerce")
            <= pd.to_datetime(selected["cutoff_time"], errors="coerce")
        )
        selected["no_out_of_window_event"] = selected["n_out_of_window_events"].eq(0)
        selected["manual_check_pass"] = (
            selected["max_not_after_cutoff"] & selected["no_out_of_window_event"]
        )
        private_rows.append(selected)
        for row in selected.itertuples(index=False):
            public_rows.append({
                "phase": phase,
                "pair_sha256": _pair_hash(row.user_id, row.course_id),
                "max_not_after_cutoff": bool(row.max_not_after_cutoff),
                "no_out_of_window_event": bool(row.no_out_of_window_event),
                "manual_check_pass": bool(row.manual_check_pass),
            })
    private = pd.concat(private_rows, ignore_index=True)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private.to_csv(private_output, index=False)
    all_pass = bool(private["manual_check_pass"].all()) and len(private) == 20
    return {
        "status": "PASS_20_PRIVATE_CASES" if all_pass else "FAIL",
        "sample_size": int(len(private)),
        "sampling": "5 rows per phase, deterministic without replacement",
        "private_checklist": str(private_output),
        "all_max_event_not_after_cutoff": bool(private["max_not_after_cutoff"].all()),
        "all_out_of_window_counts_zero": bool(private["no_out_of_window_event"].all()),
        "public_hashed_cases": public_rows,
        "limitation": "Evidence uses V3 aggregate cutoff-audit table; retain private checklist for lecturer inspection.",
    }


def _shuffle_xgb(processed_dir: Path, seed: int) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("D04 cần cài extra boosting/xgboost") from error
    X_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy", mmap_mode="r")
    shuffled = np.random.default_rng(seed).permutation(y_train)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device="cuda", random_state=seed, n_jobs=4, eval_metric="mlogloss",
    )
    model.fit(X_train, shuffled)
    per_phase = {}
    for phase_index, phase in enumerate(PHASES, start=1):
        X_dev = np.load(processed_dir / f"X_dev_phase{phase_index}.npy", mmap_mode="r")
        y_dev = np.load(processed_dir / f"y_dev_phase{phase_index}.npy", mmap_mode="r")
        pred = model.predict(X_dev).astype(int)
        per_phase[phase] = float(f1_score(y_dev, pred, labels=[0, 1, 2], average="macro", zero_division=0))
    return {"seed": seed, "learner": "xgb", "train_labels_permuted": True, "macro_f1_by_phase": per_phase, "macro_f1_mean": float(np.mean(list(per_phase.values())))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--processed-dir", help="Thư mục processed/course_quality_v1 cho D04")
    parser.add_argument("--output", default="artifacts/cq_v1_m0_probes.json")
    parser.add_argument("--target", default="y_alpha_1")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--skip-shuffle", action="store_true")
    args = parser.parse_args()
    source = Path(args.input_dir).resolve()
    feature_spec = json.loads((source / "feature_columns.json").read_text(encoding="utf-8"))
    features = list(feature_spec.get("model_feature_columns", ()))
    if not features:
        features = [*feature_spec["distance_feature_columns"], *feature_spec["categorical_features"]]
    train = pd.read_parquet(
        source / "model_ready" / "P4" / "train.parquet",
        columns=["course_id", "learner_code", "teacher_set_code", "school_set_code", args.target],
    )
    dev = pd.read_parquet(
        source / "model_ready" / "P4" / "dev.parquet",
        columns=["course_id", "learner_code", "teacher_set_code", "school_set_code", args.target],
    )
    y_train = _labels(train, args.target)
    y_dev = _labels(dev, args.target)
    majority = _majority(y_train, y_dev)
    course_probe = _categorical_probe(train, dev, ["course_id"], args.target)
    operational_probe = _categorical_probe(train, dev, ["teacher_set_code", "school_set_code"], args.target)
    learner_unseen = float((~dev["learner_code"].astype(str).isin(set(train["learner_code"].astype(str)))).mean())
    numeric_features = list(feature_spec.get("raw_numeric_features", feature_spec["distance_feature_columns"]))
    output = Path(args.output).resolve()
    private_cutoff = output.parent / "cutoff_manual_20_private.csv"
    if args.skip_shuffle:
        d04 = {"status": "SKIPPED_BY_FLAG"}
    elif not args.processed_dir:
        d04 = {"status": "BLOCKED_PROCESSED_POOL_B_REQUIRED"}
    else:
        d04 = _shuffle_xgb(Path(args.processed_dir).resolve(), args.seed)
    result = {
        "schema_version": 1,
        "split": "dev_only",
        "test_read": False,
        "D02": {
            "course_id_probe": course_probe,
            "teacher_school_probe": operational_probe,
            "new_learner_rate_dev": learner_unseen,
            "majority_reference": majority,
        },
        "D04": d04,
        "D05": {
            phase: _majority(
                _labels(pd.read_parquet(source / "model_ready" / phase / "train.parquet", columns=[args.target]), args.target),
                _labels(pd.read_parquet(source / "model_ready" / phase / "dev.parquet", columns=[args.target]), args.target),
            )
            for phase in PHASES
        },
        "D03": _missingness_probe(source, args.target, numeric_features, args.seed),
        "D06": _cutoff_probe(source, private_cutoff, args.seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "D02_course_macro_f1": course_probe["macro_f1"], "D02_majority_macro_f1": majority["macro_f1"], "D04": result["D04"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
