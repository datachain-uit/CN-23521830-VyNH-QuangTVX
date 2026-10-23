"""Tạo data-card và bảng kiểm M0 CQ v1 mà không mở test metric.

Script chỉ đọc metadata/nhãn để kiểm toán giao thức. Nó không ghi danh sách ID
ra artifact công khai: split manifest chỉ chứa số lượng và SHA-256 của ánh xạ.
Các kiểm tra cần raw event, FullTrain-A hoặc xác nhận ``course_root_id`` được ghi
BLOCKED thay vì suy đoán là PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASES = ("P1", "P2", "P3", "P4")
SPLITS = ("train", "dev", "test")
CLASSES = ("E", "G", "A")
TARGETS = {"afelo1": ("q_alpha_1", "y_alpha_1"), "afelo0p1": ("q_alpha_0p1", "y_alpha_0p1")}


def _quality_class(q: pd.Series) -> np.ndarray:
    values = q.to_numpy(dtype=float)
    return np.where(values < 0.2, "E", np.where(values < 0.4, "G", "A"))


def _hash_rows(frame: pd.DataFrame) -> str:
    rows = frame.astype(str).sort_values(list(frame.columns), kind="mergesort")
    payload = "\n".join("\t".join(row) for row in rows.to_numpy()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _icc1(values: pd.DataFrame) -> float | None:
    grouped = values.groupby("course_id")["q"].agg(["size", "mean"])
    if len(grouped) < 2 or int(grouped["size"].sum()) <= len(grouped):
        return None
    grand = float(values["q"].mean())
    ss_between = float((grouped["size"] * (grouped["mean"] - grand) ** 2).sum())
    merged = values.join(grouped["mean"].rename("course_mean"), on="course_id")
    ss_within = float(((merged["q"] - merged["course_mean"]) ** 2).sum())
    df_between = len(grouped) - 1
    df_within = len(values) - len(grouped)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    n_bar = (len(values) - float((grouped["size"] ** 2).sum()) / len(values)) / df_between
    denominator = ms_between + (n_bar - 1.0) * ms_within
    return float((ms_between - ms_within) / denominator) if denominator else None


def _distribution(frame: pd.DataFrame, q_col: str, y_col: str, n_min: int) -> dict[str, Any]:
    enrollments = frame[["user_id", "course_id", q_col, y_col]].rename(columns={q_col: "q", y_col: "y"})
    course_n = enrollments.groupby("course_id").size()
    course_q = enrollments.groupby("course_id")["q"].mean()
    course_y = pd.Series(_quality_class(course_q), index=course_q.index)
    per_course_prior = (
        pd.crosstab(enrollments["course_id"], enrollments["y"], normalize="index")
        .reindex(columns=CLASSES, fill_value=0.0)
    )
    quantiles = course_n.quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
    return {
        "rows": int(len(enrollments)),
        "users": int(enrollments["user_id"].nunique()),
        "courses": int(enrollments["course_id"].nunique()),
        "enrollment_label_counts": {label: int((enrollments["y"] == label).sum()) for label in CLASSES},
        "course_label_counts": {label: int((course_y == label).sum()) for label in CLASSES},
        "n_learners_per_course": {str(key): float(value) for key, value in quantiles.items()},
        "courses_below_n_min": int((course_n < n_min).sum()),
        "icc1_q_by_course": _icc1(enrollments),
        "course_prior_std": {label: float(per_course_prior[label].std(ddof=1)) for label in CLASSES},
    }


def build_audit(
    source: Path,
    *,
    target: str,
    n_min: int,
    root_policy: str,
    probe_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    q_col, y_col = TARGETS[target]
    source_validation = json.loads((source / "validation.json").read_text(encoding="utf-8"))
    feature_spec = json.loads((source / "feature_columns.json").read_text(encoding="utf-8"))
    summary_path = source / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = json.loads((source / "_SUCCESS.json").read_text(encoding="utf-8"))
    available_splits = tuple(
        split for split in SPLITS
        if all((source / "model_ready" / phase / f"{split}.parquet").is_file() for phase in PHASES)
    )
    if not {"train", "dev"} <= set(available_splits):
        raise FileNotFoundError("Data card M0 cần đủ train/dev P1--P4")
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    required = ["user_id", "course_id", "phase", "split", "enroll_time", "cutoff_time", q_col, y_col]
    for phase in PHASES:
        for split in available_splits:
            path = source / "model_ready" / phase / f"{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path, columns=required)
            frames[(phase, split)] = frame

    p1 = {split: frames[("P1", split)] for split in available_splits}
    split_pairs = pd.concat(
        [frame[["course_id"]].drop_duplicates().assign(split=split) for split, frame in p1.items()],
        ignore_index=True,
    )
    split_overlap = int((split_pairs.groupby("course_id")["split"].nunique() > 1).sum())
    label_formula_ok = all(
        np.array_equal(_quality_class(frame[q_col]), frame[y_col].astype(str).to_numpy())
        for frame in frames.values()
    )
    unique_grain_ok = all(
        not frame.duplicated(["user_id", "course_id", "phase"]).any()
        for frame in frames.values()
    )
    same_keys_all_phases = all(
        _hash_rows(frames[(phase, split)][["user_id", "course_id"]])
        == _hash_rows(p1[split][["user_id", "course_id"]])
        for phase in PHASES for split in available_splits
    )
    train = p1["train"][["course_id", q_col, y_col]].rename(columns={q_col: "q", y_col: "y"})
    mu = {label: float(train.loc[train["y"].astype(str) == label, "q"].mean()) for label in CLASSES}
    mu_valid = mu["E"] < mu["G"] < mu["A"] and mu["E"] < 0.2 <= mu["G"] < 0.4 <= mu["A"] <= 1.0

    dev = p1["dev"][["course_id", q_col, y_col]].rename(columns={q_col: "q", y_col: "y"})
    dev_course = dev.groupby("course_id").agg(Q_true=("q", "mean"), n_learners=("q", "size"))
    dev_qhat = dev.assign(mu_y=dev["y"].astype(str).map(mu)).groupby("course_id")["mu_y"].mean()
    eligible = dev_course[dev_course["n_learners"] >= n_min].join(dev_qhat)
    oracle_floor = float((eligible["mu_y"] - eligible["Q_true"]).abs().mean()) if len(eligible) else None

    target_columns = set(feature_spec.get("target_columns", []))
    x_columns = set(feature_spec.get("distance_feature_columns", []))
    selector_columns = x_columns - {"phase_fraction"}
    learner_columns = set(feature_spec.get("model_feature_columns", ())) or (
        x_columns | set(feature_spec.get("categorical_features", ()))
    )
    forbidden_in_x = sorted(target_columns & x_columns)
    source_false = sorted(key for key, value in source_validation.items() if value is False)
    root_confirmed = root_policy == "course_id_equals_root"
    probe_evidence = probe_evidence or {}
    d02 = probe_evidence.get("D02", {})
    course_probe = d02.get("course_id_probe", {})
    majority_probe = d02.get("majority_reference", {})
    d02_pass = (
        course_probe.get("unseen_rate_dev", {}).get("course_id") == 1.0
        and abs(float(course_probe.get("macro_f1", -1)) - float(majority_probe.get("macro_f1", -2))) <= 1e-12
    )
    d04_probe = probe_evidence.get("D04", {})
    d04_status = "PASS" if isinstance(d04_probe.get("macro_f1_mean"), (int, float)) else "BLOCKED"
    d05_pass = set(probe_evidence.get("D05", {})) == set(PHASES)
    d03_probe = probe_evidence.get("D03", {})
    d03_pass = d03_probe.get("status") == "PASS_PROBE_COMPLETED"
    d06_probe = probe_evidence.get("D06", {})
    d06_pass = d06_probe.get("status") == "PASS_20_PRIVATE_CASES"
    split_variant = str(feature_spec.get("split_variant", "unknown"))
    split_protocol_approved = split_variant in {"temporal", "sgkf"}
    checks = {
        "D01_size_distribution_split": {"status": "PASS", "evidence": "rows/users/courses/classes emitted below"},
        "D02_id_only_probe": {
            "status": "PASS" if d02_pass else "BLOCKED",
            "reason": "course-only dev Macro-F1 equals majority and all dev course IDs are unseen" if d02_pass else "run scripts/run_cq_v1_m0_probes.py",
            "evidence": d02,
        },
        "D03_missingness_probe": {"status": "PASS" if d03_pass else "BLOCKED", "reason": "pre-imputation missing-mask probe completed; masks excluded from production X" if d03_pass else "raw pre-imputation matrix is required", "evidence": d03_probe},
        "D04_shuffled_labels": {"status": d04_status, "reason": "XGBoost shuffled-label dev probe" if d04_status == "PASS" else "requires FullTrain-B development run"},
        "D05_majority_baseline": {"status": "PASS" if d05_pass else "READY", "reason": "majority Macro-F1 emitted for P1-P4" if d05_pass else "class counts available; run probe script"},
        "D06_cutoff_manual_20": {"status": "PASS" if d06_pass else "BLOCKED", "reason": "20 private cases (5/phase) have max event <= cutoff and zero out-of-window events" if d06_pass else "requires 20 private timestamp cases", "evidence": d06_probe},
        "D07_weight_contract": {"status": "PASS_CODE", "reason": "covered by unit tests; run evidence still required"},
        "D08_baseline_fidelity": {"status": "DEFERRED_M1", "reason": "review explicitly allows completion before M1"},
        "D09_unit_label": {"status": "PASS" if unique_grain_ok and label_formula_ok and same_keys_all_phases else "FAIL"},
        "D10_no_label_in_X": {"status": "PASS" if not forbidden_in_x else "FAIL", "forbidden_in_X": forbidden_in_x},
        "D11_course_recompute": {"status": "PASS_GENERATED", "reason": "Qc/Yc recomputed from enrollment q; this audit creates the first data-card values"},
        "D12_protocol_constants": {"status": "PASS" if mu_valid else "FAIL", "mu": mu},
        "course_root_contract": {
            "status": "PASS" if root_confirmed else "BLOCKED",
            "reason": "course_id confirmed as one opening/root" if root_confirmed else "awaiting written confirmation that each course_id opens once",
        },
        "split_protocol_contract": {
            "status": "PASS" if split_protocol_approved else "BLOCKED_APPROVAL",
            "reason": f"source split_variant={split_variant}; review requested temporal/SGKF unless lecturer approves deviation",
        },
    }
    return {
        "schema_version": 1,
        "dataset_version": summary.get("version"),
        "target": {"id": target, "q_column": q_col, "y_column": y_col, "cuts": [0.2, 0.4], "lower_is_better": True},
        "n_min": n_min,
        "delta_c": 0.01,
        "root_policy": root_policy,
        "confirmatory_eligible": root_confirmed and split_protocol_approved and not source_false and split_overlap == 0 and "test" in available_splits,
        "available_splits": list(available_splits),
        "test_data_packaged": "test" in available_splits,
        "source_validation_false": source_false,
        "grain": ["user_id", "course_id", "phase"],
        "four_phases_same_enrollment_keys": same_keys_all_phases,
        "split_course_overlap": split_overlap,
        "split_manifest_public": {
            "mapping_sha256": _hash_rows(split_pairs),
            "actual_ids_published": False,
            "group_unit": "course_id_confirmed_as_root" if root_confirmed else "course_id_provisional",
        },
        "distribution_by_split": {split: _distribution(p1[split], q_col, y_col, n_min) for split in available_splits},
        "protocol_constants_train": mu,
        "dev_oracle_floor_ec": oracle_floor,
        "dev_oracle_floor_eligible_courses": int(len(eligible)),
        "feature_contract": {
            "selector_feature_count": len(selector_columns),
            "learner_feature_count": len(learner_columns),
            "learner_static_code_columns": sorted(set(feature_spec.get("categorical_features", ())) & learner_columns),
            "target_columns": sorted(target_columns),
            "target_columns_in_selector_X": forbidden_in_x,
            "missing_indicators_added": any(name.startswith("missing_") for name in learner_columns),
        },
        "checks": checks,
    }


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# CQ v1 — Data card và kiểm tra M0 (provisional)", "",
        "> Artifact tự sinh từ `scripts/audit_cq_v1_m0.py`. Không chứa ID thật và không dùng dự đoán test.", "",
        f"- Dataset version: `{audit['dataset_version']}`",
        f"- Target: `{audit['target']['y_column']}` từ `{audit['target']['q_column']}`; ngưỡng 0,2/0,4; giá trị nhỏ là tốt.",
        f"- `n_min={audit['n_min']}`, `delta_C={audit['delta_c']}`.",
        f"- Group unit: `{audit['split_manifest_public']['group_unit']}`.",
        f"- Confirmatory eligible: `{str(audit['confirmatory_eligible']).lower()}`.", "",
        "## Hai feature view theo O17/QĐ3", "",
        f"- Selector view: `{audit['feature_contract']['selector_feature_count']}` feature hành vi; không chứa cột mã.",
        f"- Learner view: `{audit['feature_contract']['learner_feature_count']}` feature; giữ `{', '.join(audit['feature_contract']['learner_static_code_columns'])}` theo TEMPO-DQ.",
        "- `course_id`, target và target-derived columns không nằm trong hai view.", "",
        "## Phân bố", "",
        "| Split | Enrollment | User | Course | E | G | A | Course < n_min | ICC(1) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in audit["distribution_by_split"].items():
        counts = row["enrollment_label_counts"]
        icc = "NA" if row["icc1_q_by_course"] is None else f"{row['icc1_q_by_course']:.6f}"
        lines.append(f"| {split} | {row['rows']} | {row['users']} | {row['courses']} | {counts['E']} | {counts['G']} | {counts['A']} | {row['courses_below_n_min']} | {icc} |")
    lines.extend(["", "## Hằng số train và sàn EC dev", ""])
    lines.append("- " + ", ".join(f"mu_{key}={value:.8f}" for key, value in audit["protocol_constants_train"].items()))
    lines.append(f"- Oracle floor EC dev (n_min): `{audit['dev_oracle_floor_ec']}` trên `{audit['dev_oracle_floor_eligible_courses']}` khóa.")
    lines.extend(["", "## Checks 1–12", "", "| Check | Status | Lý do/evidence |", "|---|---|---|"])
    for key, row in audit["checks"].items():
        detail = row.get("reason") or row.get("evidence") or ""
        lines.append(f"| {key} | {row['status']} | {detail} |")
    lines.extend(["", "## Giới hạn chưa được phép suy đoán", "", "- Chưa xác nhận `course_id == course_root_id` bằng văn bản.", "- Split `course_random` cần giảng viên chấp thuận deviation so với temporal/SGKF đã đăng ký.", "- Chưa có FullTrain-A (full-behavior) trong snapshot Pool B.", "- Test không nằm trong gói M0; không có fit, prediction hay metric test.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="docs/cq_v1")
    parser.add_argument("--target", choices=sorted(TARGETS), default="afelo1")
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--root-policy", choices=("pending", "course_id_equals_root"), default="pending")
    parser.add_argument("--probe-evidence")
    args = parser.parse_args()
    source = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe_evidence = None
    if args.probe_evidence:
        probe_evidence = json.loads(Path(args.probe_evidence).read_text(encoding="utf-8"))
    audit = build_audit(
        source, target=args.target, n_min=args.n_min,
        root_policy=args.root_policy, probe_evidence=probe_evidence,
    )
    (output / "m0_data_card_provisional.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "M0_DATA_CARD_PROVISIONAL.md").write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps({"output": str(output), "confirmatory_eligible": audit["confirmatory_eligible"], "checks": {k: v["status"] for k, v in audit["checks"].items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
