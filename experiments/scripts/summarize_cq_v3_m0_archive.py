"""Kiểm toán và tóm tắt archive CQ V3 M0 mà không bung dữ liệu riêng tư.

Script chỉ đọc member trực tiếp từ ``tar.gz``. Nó từ chối archive có dấu vết
test, kiểm tra trạng thái cell, các check D01--D12, đủ ô FullTrain-B/headroom,
rồi xuất JSON và Markdown để ghép vào báo cáo xin GO. Kết luận của script chỉ
có thể là ``READY_TO_REQUEST_CONDITIONAL_GO``; các phê duyệt group/split,
FullTrain-A và QĐ16 vẫn thuộc thẩm quyền giảng viên.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any


EXPECTED_COMMIT = "81e1181a8ca25b0dceb9c387a3e3811a6f2d3466"
EXPECTED_PHASES = {"dev_phase1", "dev_phase2", "dev_phase3", "dev_phase4"}
EXPECTED_FULLTRAIN_POLICIES = {"balanced", "natural"}
EXPECTED_LEARNERS = {"lr", "xgb"}
EXPECTED_HEADROOM_RATIOS = {0.005, 0.01, 0.02, 0.05}
EXPECTED_SELECTOR_SEEDS = {11, 29, 47}
LABEL_NAMES = {"0": "E", "1": "G", "2": "A"}
OVERALL_FIELDS = (
    "accuracy", "balanced_accuracy", "mcc", "f1_macro", "worst_class_f1",
    "worst_class_recall", "roc_auc_macro", "pr_auc_macro", "log_loss",
    "brier_multiclass", "ece_15",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArchiveReader:
    """Đọc member duy nhất theo suffix và phát hiện member trùng."""

    def __init__(self, path: Path):
        self.path = path
        self.tar = tarfile.open(path, "r:gz")
        self.members = [member for member in self.tar.getmembers() if member.isfile()]

    def close(self) -> None:
        self.tar.close()

    def names(self) -> list[str]:
        return [member.name for member in self.members]

    def matching(self, suffix: str) -> list[tarfile.TarInfo]:
        return [member for member in self.members if member.name.endswith(suffix)]

    def bytes(self, member: tarfile.TarInfo) -> bytes:
        stream = self.tar.extractfile(member)
        if stream is None:
            raise ValueError(f"Không đọc được member {member.name}")
        return stream.read()

    def json(self, suffix: str, *, required: bool = True) -> dict[str, Any] | None:
        matches = self.matching(suffix)
        if not matches and not required:
            return None
        if len(matches) != 1:
            raise ValueError(f"Cần đúng một *{suffix}, tìm thấy {len(matches)}")
        return json.loads(self.bytes(matches[0]).decode("utf-8"))

    def text(self, suffix: str, *, required: bool = True) -> str | None:
        matches = self.matching(suffix)
        if not matches and not required:
            return None
        if len(matches) != 1:
            raise ValueError(f"Cần đúng một *{suffix}, tìm thấy {len(matches)}")
        return self.bytes(matches[0]).decode("utf-8")


def _forbidden_test_members(names: list[str]) -> list[str]:
    patterns = (
        re.compile(r"(^|/)test\.parquet$", re.IGNORECASE),
        re.compile(r"(^|/)eval-test(?:/|$)", re.IGNORECASE),
        re.compile(r"(^|/)test_predictions(?:\.|/|$)", re.IGNORECASE),
        re.compile(r"(^|/)metrics_test(?:\.|/|$)", re.IGNORECASE),
    )
    return sorted(name for name in names if any(pattern.search(name) for pattern in patterns))


def _parse_eval_path(name: str) -> dict[str, Any] | None:
    parts = PurePosixPath(name).parts
    if "experiments" not in parts or not name.endswith("/metrics_overall.json"):
        return None
    eval_part = next((part for part in parts if part.startswith("eval-")), None)
    learner = next((part for part in parts if part in EXPECTED_LEARNERS), None)
    selector = next((part for part in parts if part.startswith("ss") and part[2:].isdigit()), None)
    budget = next((part for part in parts if re.fullmatch(r"b\d+p\d+", part)), None)
    method = next((part for part in parts if part in {"c00_full_train", "c02_stratified_random"}), None)
    if not all((eval_part, learner, selector, budget, method)):
        return None
    experiment_index = parts.index("experiments") + 1
    experiment = parts[experiment_index]
    phase = eval_part.removeprefix("eval-")
    return {
        "experiment_id": experiment,
        "method_id": method,
        "budget_token": budget,
        "selector_seed": int(selector[2:]),
        "learner_id": learner,
        "phase": phase,
    }


def _policy(experiment_id: str) -> str | None:
    for value in EXPECTED_FULLTRAIN_POLICIES:
        if f"_{value}_v1" in experiment_id:
            return value
    return None


def _read_evaluations(reader: ArchiveReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for member in reader.members:
        keys = _parse_eval_path(member.name)
        if keys is None:
            continue
        metrics = json.loads(reader.bytes(member).decode("utf-8"))
        rows.append({**keys, **{field: metrics.get(field) for field in OVERALL_FIELDS}})
    return rows


def _read_per_label(reader: ArchiveReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for member in reader.members:
        if not member.name.endswith("/metrics_per_label.csv"):
            continue
        keys = _parse_eval_path(member.name.replace("metrics_per_label.csv", "metrics_overall.json"))
        if keys is None:
            continue
        text = reader.bytes(member).decode("utf-8-sig")
        for record in csv.DictReader(io.StringIO(text)):
            rows.append({**keys, **record})
    return rows


def _fulltrain_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    selected = [
        row for row in rows
        if row["method_id"] == "c00_full_train"
        and row["experiment_id"].startswith("cq_v3_m0_fulltrain_b_dev_")
        and row["experiment_id"] != "cq_v3_m0_fulltrain_b_dev_natural_xgb_v1"
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selected:
        policy = _policy(row["experiment_id"])
        if policy:
            groups.setdefault((policy, row["learner_id"]), []).append(row)
    summary = []
    errors = []
    for policy in sorted(EXPECTED_FULLTRAIN_POLICIES):
        for learner in sorted(EXPECTED_LEARNERS):
            values = groups.get((policy, learner), [])
            phases = {row["phase"] for row in values}
            if len(values) != 4 or phases != EXPECTED_PHASES:
                errors.append(f"FullTrain-B {policy}/{learner}: cần 4 phase, có {len(values)} {sorted(phases)}")
                continue
            item: dict[str, Any] = {"policy": policy, "learner_id": learner, "phase_count": 4}
            for field in OVERALL_FIELDS:
                numeric = [float(row[field]) for row in values if isinstance(row.get(field), (int, float))]
                item[f"{field}_mean"] = mean(numeric) if len(numeric) == 4 else None
            summary.append(item)
    return summary, errors


def _per_label_summary(
    rows: list[dict[str, Any]], evaluations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    expected_keys = {
        (row["experiment_id"], row["method_id"], row["budget_token"], row["selector_seed"], row["learner_id"], row["phase"])
        for row in evaluations
    }
    counts = Counter(
        (row["experiment_id"], row["method_id"], row["budget_token"], row["selector_seed"], row["learner_id"], row["phase"])
        for row in rows
    )
    incomplete = {key: counts.get(key, 0) for key in expected_keys if counts.get(key, 0) != 3}
    if incomplete:
        preview = list(sorted(incomplete.items()))[:5]
        errors.append(f"Per-label cần đúng 3 lớp cho mỗi eval; lỗi đầu: {preview}")
    selected = [
        row for row in rows
        if row["method_id"] == "c00_full_train"
        and row["experiment_id"].startswith("cq_v3_m0_fulltrain_b_dev_")
        and row["experiment_id"] != "cq_v3_m0_fulltrain_b_dev_natural_xgb_v1"
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in selected:
        policy = _policy(row["experiment_id"])
        label = LABEL_NAMES.get(str(row.get("label")))
        if policy and label:
            groups.setdefault((policy, row["learner_id"], label), []).append(row)
    summary: list[dict[str, Any]] = []
    fields = ("prevalence", "precision", "recall", "specificity", "f1", "f2", "mcc", "roc_auc", "pr_auc")
    for policy in sorted(EXPECTED_FULLTRAIN_POLICIES):
        for learner in sorted(EXPECTED_LEARNERS):
            for label in ("E", "G", "A"):
                values = groups.get((policy, learner, label), [])
                phases = {row["phase"] for row in values}
                if len(values) != 4 or phases != EXPECTED_PHASES:
                    errors.append(f"Per-label FullTrain-B {policy}/{learner}/{label}: cần 4 phase, có {len(values)}")
                    continue
                item: dict[str, Any] = {"policy": policy, "learner_id": learner, "label": label}
                for field in fields:
                    numeric = [float(row[field]) for row in values if row.get(field) not in (None, "")]
                    item[f"{field}_mean"] = mean(numeric) if len(numeric) == 4 else None
                summary.append(item)
    return summary, errors


def _headroom_completeness(rows: list[dict[str, Any]], decision: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    c02 = [row for row in rows if row["experiment_id"].startswith("cq_v3_m0_headroom_c02_")]
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in c02:
        by_experiment.setdefault(row["experiment_id"], []).append(row)
    expected_experiments = 4
    if len(by_experiment) != expected_experiments:
        errors.append(f"Headroom C02: cần {expected_experiments} config, có {len(by_experiment)}")
    for experiment, values in sorted(by_experiment.items()):
        phases = {row["phase"] for row in values}
        seeds = {row["selector_seed"] for row in values}
        if len(values) != 12 or phases != EXPECTED_PHASES or seeds != EXPECTED_SELECTOR_SEEDS:
            errors.append(
                f"{experiment}: cần 12 eval = 4 phase x 3 seed; có {len(values)}, "
                f"phase={sorted(phases)}, seed={sorted(seeds)}"
            )
    full = [row for row in rows if row["experiment_id"] == "cq_v3_m0_fulltrain_b_dev_natural_xgb_v1"]
    if len(full) != 4 or {row["phase"] for row in full} != EXPECTED_PHASES:
        errors.append(f"Headroom FullTrain-B natural/XGB: cần 4 phase, có {len(full)}")
    ratios = set()
    if decision:
        ratios = {float(value) for value in decision.get("ratios", {})}
        if ratios != EXPECTED_HEADROOM_RATIOS:
            errors.append(f"Headroom decision sai ratio: {sorted(ratios)}")
        if decision.get("evidence_split") != "dev" or decision.get("test_data_read") is not False:
            errors.append("Headroom decision không chứng minh dev-only")
    else:
        errors.append("Thiếu headroom decision")
    return {
        "c02_evaluation_count": len(c02),
        "c02_config_count": len(by_experiment),
        "fulltrain_evaluation_count": len(full),
        "ratios": sorted(ratios),
        "rho_star": decision.get("rho_star") if decision else None,
        "fallback_to_smallest": decision.get("fallback_to_smallest") if decision else None,
    }, errors


def summarize(path: Path) -> dict[str, Any]:
    reader = ArchiveReader(path)
    try:
        names = reader.names()
        status = reader.json("cq_v3_m0_execution_status.json", required=False) or {}
        commit = (reader.text("repository_commit_cq_v3_m0.txt", required=False) or "").strip()
        probes = reader.json("m0_docs/cq_v3/m0_probes_dev_summary.json", required=False) or {}
        data_card = reader.json("m0_docs/cq_v3/m0_data_card_provisional.json", required=False) or {}
        headroom = reader.json("m0_docs/cq_v3/m0_headroom_decision_cq_v3.json", required=False)
        evaluations = _read_evaluations(reader)
        per_label = _read_per_label(reader)
    finally:
        reader.close()

    forbidden = _forbidden_test_members(names)
    step_failures = {
        name: row for name, row in status.get("steps", {}).items()
        if row.get("status") != "success"
    }
    checks = data_card.get("checks", {})
    required_check_names = {
        "D01_size_distribution_split", "D02_id_only_probe", "D03_missingness_probe",
        "D04_shuffled_labels", "D05_majority_baseline", "D06_cutoff_manual_20",
        "D07_weight_contract", "D09_unit_label", "D10_no_label_in_X",
        "D11_course_recompute", "D12_protocol_constants",
    }
    check_failures = {
        name: checks.get(name, {}).get("status", "MISSING")
        for name in sorted(required_check_names)
        if checks.get(name, {}).get("status") not in {"PASS", "PASS_CODE", "PASS_GENERATED"}
    }
    fulltrain, fulltrain_errors = _fulltrain_summary(evaluations)
    per_label_summary, per_label_errors = _per_label_summary(per_label, evaluations)
    headroom_summary, headroom_errors = _headroom_completeness(evaluations, headroom)
    privacy_pass = (
        not forbidden
        and status.get("test_files_permitted") is False
        and status.get("test_metrics_opened") is False
        and data_card.get("test_data_packaged") is False
    )
    archive_success = status.get("status") == "success_conditional_go_evidence_ready"
    code_pin_pass = commit == EXPECTED_COMMIT == status.get("repository_commit")
    evidence_complete = not (
        step_failures or check_failures or fulltrain_errors or per_label_errors or headroom_errors
    )
    request_status = (
        "READY_TO_REQUEST_CONDITIONAL_GO"
        if archive_success and privacy_pass and code_pin_pass and evidence_complete
        else "NOT_READY"
    )
    external_blockers = {
        "course_root_contract": checks.get("course_root_contract", {}).get("status", "MISSING"),
        "split_protocol_contract": checks.get("split_protocol_contract", {}).get("status", "MISSING"),
        "fulltrain_a": status.get("fulltrain_a_status", "MISSING"),
        "qd16": "PARTIAL_9_OF_13_REQUIRES_SEPARATE_EVIDENCE",
    }
    return {
        "schema_version": 1,
        "archive": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256(path),
        "request_status": request_status,
        "full_go": False,
        "full_go_reason": "requires written group/split/FullTrain-A decisions and complete QD16 evidence",
        "archive_execution_status": status.get("status"),
        "repository_commit": commit,
        "code_pin_pass": code_pin_pass,
        "privacy": {
            "pass": privacy_pass,
            "forbidden_members": forbidden,
            "test_files_permitted": status.get("test_files_permitted"),
            "test_metrics_opened": status.get("test_metrics_opened"),
            "test_data_packaged": data_card.get("test_data_packaged"),
        },
        "step_failures": step_failures,
        "mandatory_check_failures": check_failures,
        "probe_status": {key: value.get("status") if isinstance(value, dict) else None for key, value in probes.items()},
        "fulltrain_summary": fulltrain,
        "fulltrain_errors": fulltrain_errors,
        "per_label_summary": per_label_summary,
        "per_label_errors": per_label_errors,
        "headroom": headroom_summary,
        "headroom_errors": headroom_errors,
        "evaluation_rows": len(evaluations),
        "per_label_rows": len(per_label),
        "external_blockers": external_blockers,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CQ V3 M0 — kiểm toán archive Kaggle", "",
        f"- Trạng thái xin GO có điều kiện: **{summary['request_status']}**.",
        f"- Full GO: **NO** — {summary['full_go_reason']}.",
        f"- SHA-256 archive: `{summary['archive_sha256']}`.",
        f"- Commit code: `{summary['repository_commit']}`; pin hợp lệ: `{str(summary['code_pin_pass']).lower()}`.",
        f"- Dev-only/privacy: `{str(summary['privacy']['pass']).lower()}`.", "",
        "## FullTrain-B dev", "",
        "| Class weighting | Learner | Macro-F1 | MCC | Worst-class F1 | Worst-class recall | PR-AUC | Log loss | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fulltrain_summary"]:
        lines.append(
            f"| {row['policy']} | {row['learner_id']} | {_fmt(row['f1_macro_mean'])} | "
            f"{_fmt(row['mcc_mean'])} | {_fmt(row['worst_class_f1_mean'])} | "
            f"{_fmt(row['worst_class_recall_mean'])} | {_fmt(row['pr_auc_macro_mean'])} | "
            f"{_fmt(row['log_loss_mean'])} | {_fmt(row['ece_15_mean'])} |"
        )
    lines.extend([
        "", "## FullTrain-B theo lớp (trung bình P1--P4)", "",
        "| Class weighting | Learner | Lớp | Prevalence | Precision | Recall | Specificity | F1 | PR-AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["per_label_summary"]:
        lines.append(
            f"| {row['policy']} | {row['learner_id']} | {row['label']} | "
            f"{_fmt(row['prevalence_mean'])} | {_fmt(row['precision_mean'])} | "
            f"{_fmt(row['recall_mean'])} | {_fmt(row['specificity_mean'])} | "
            f"{_fmt(row['f1_mean'])} | {_fmt(row['pr_auc_mean'])} |"
        )
    lines.extend([
        "", "## Headroom QĐ12", "",
        f"- rho-star: `{summary['headroom']['rho_star']}`.",
        f"- C02 evaluations: `{summary['headroom']['c02_evaluation_count']}`; "
        f"FullTrain reference evaluations: `{summary['headroom']['fulltrain_evaluation_count']}`.",
        f"- Fallback về budget nhỏ nhất: `{summary['headroom']['fallback_to_smallest']}`.", "",
        "## Kiểm tra lỗi", "",
        f"- Step lỗi/thiếu: `{json.dumps(summary['step_failures'], ensure_ascii=False)}`.",
        f"- Check D lỗi/thiếu: `{json.dumps(summary['mandatory_check_failures'], ensure_ascii=False)}`.",
        f"- FullTrain completeness: `{json.dumps(summary['fulltrain_errors'], ensure_ascii=False)}`.",
        f"- Per-label completeness: `{json.dumps(summary['per_label_errors'], ensure_ascii=False)}`.",
        f"- Headroom completeness: `{json.dumps(summary['headroom_errors'], ensure_ascii=False)}`.", "",
        "## Blocker ngoài archive", "",
    ])
    for key, value in summary["external_blockers"].items():
        lines.append(f"- `{key}`: `{value}`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output-dir", default="artifacts/cq_v3_m0_archive_summary")
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = summarize(archive)
    (output / "m0_archive_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "M0_ARCHIVE_SUMMARY.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "request_status": result["request_status"],
        "full_go": result["full_go"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if result["request_status"] == "READY_TO_REQUEST_CONDITIONAL_GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
