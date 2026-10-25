"""Reanalyse existing V20/V50 and learner-code DEV predictions for Y32/QD32.

CPU only: no fit, no model loading, no test input, no source overwrite.
The source is the locked V4 runtime_dev_only bundle; row/label/family alignment
is checked before computing new aggregates. Original archives stay immutable.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from skq_exp.artifacts import sha256_file
from skq_exp.data.adapters.course_quality_v4 import _verify_sha256s
from skq_exp.data.cq_patch_1023 import calibration_in_the_large, subgroup_log_loss, row_log_loss, course_contributions
from skq_exp.experiments.cq_review_1025 import e_phase_report, probabilities
from skq_exp.experiments.cq_decisions_1023 import paired_family_logloss_ci
if __package__:
    from .compute_cq_v4_dev_endpoints import align_predictions
else:
    from compute_cq_v4_dev_endpoints import align_predictions

PHASES = ("P1", "P2", "P3", "P4")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_runs(path, dev):
    runs = {}
    with tarfile.open(path) as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        if len(members) != sum(m.isfile() for m in archive.getmembers()):
            raise ValueError("Duplicate archive members")
        if any("eval-test" in n.lower() or ".." in Path(n).parts for n in members):
            raise ValueError("Test/path traversal artifact rejected")
        read = lambda n: archive.extractfile(members[n]).read()
        for name in members:
            if not name.endswith("/manifest.json") or "/experiments/" not in name:
                continue
            manifest = json.loads(read(name))
            split = manifest.get("evaluation_split", "")
            if split not in dev:
                continue
            if manifest["dataset_id"] != "course_quality_v4" or manifest["model_seed"] != 42:
                raise ValueError("Expected CQ V4 model seed42")
            phase = "P" + split[-1]
            key = manifest["experiment_id"]
            if phase in runs.setdefault(key, {}):
                raise ValueError("Mixed fitted models or duplicate phase")
            stem = name.removesuffix("manifest.json")
            with np.load(io.BytesIO(read(stem + "predictions.npz")), allow_pickle=False) as data:
                pred = align_predictions(dev[split], dict(data))
            cost = json.loads(read(stem + "cost.json"))
            metrics = json.loads(read(stem + "metrics_all.json")) if stem + "metrics_all.json" in members else {}
            runs[key][phase] = dict(pred=pred, cost=cost, metrics=metrics,
                prediction_sha256=__import__('hashlib').sha256(read(stem + "predictions.npz")).hexdigest())
    if not runs or any(set(parts) != set(PHASES) for parts in runs.values()):
        raise ValueError("Missing complete dev prediction run")
    return runs


def table(headers, rows):
    def fmt(v):
        return "—" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers),
                       *["| " + " | ".join(map(fmt, row)) + " |" for row in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "v20", "v50", "learner-ab", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh report directory required")
    _verify_sha256s(args.source)
    train_users = set(pd.read_parquet(args.source / "pool_b/train.parquet", columns=["user_id"]).user_id)
    dev = {f"dev_phase{i}": pd.read_parquet(args.source / f"pool_b/dev_phase{i}.parquet",
           columns=["row_id", "user_id", "course_id", "course_family_id", "phase", "y_alpha_1"])
           for i in range(1, 5)}
    originals = {"V20": args.v20, "V50": args.v50, "learner_ab": args.learner_ab}
    report = {"review": "2026-10-25", "test_data_read": False, "new_model_fits": 0,
              "source_bundle_manifest_sha256": sha256_file(args.source / "SHA256SUMS.txt"),
              "input_archives": {k: {"file": v.name, "sha256": sha256_file(v)} for k, v in originals.items()},
              "runs": {}, "teacher_subgroup": "pending_data_team_mapping"}
    private = {}
    for source_name, path in originals.items():
        for run_id, parts in read_runs(path, dev).items():
            print(f"ANALYZE {source_name}/{run_id}: aligned dev; computing E family CI", flush=True)
            p = {ph: probabilities(parts[ph]["pred"]["y_prob"]) for ph in PHASES}
            y = {ph: parts[ph]["pred"]["y_true"] for ph in PHASES}
            fam = {ph: dev[f"dev_phase{ph[-1]}"].course_family_id.to_numpy() for ph in PHASES}
            per, ll, calibration, subgroup, e_by_user, fp = {}, {}, {}, {}, {}, {}
            for ph in PHASES:
                frame = dev[f"dev_phase{ph[-1]}"]
                seen = frame.user_id.isin(train_users).to_numpy()
                group = np.where(seen, "seen", "new")
                ll[ph] = float(row_log_loss(p[ph], y[ph]).mean())
                calibration[ph] = calibration_in_the_large(p[ph], y[ph])
                subgroup[ph] = subgroup_log_loss(p[ph], y[ph], group)
                e_by_user[ph] = {g: calibration_in_the_large(p[ph][group == g], y[ph][group == g])["E"]
                                 for g in ("seen", "new") if np.any(group == g)}
                mask = (p[ph].argmax(1) == 0) & (y[ph] != 0)
                fp[ph] = {"false_positive_E": int(mask.sum()),
                           "new_user_false_positive_E": int((mask & ~seen).sum()),
                           "new_user_fraction_in_false_positive_E": float((mask & ~seen).sum()/mask.sum()) if mask.any() else None}
                curve = parts[ph]["metrics"].get("round_diagnostics", {}).get("points", {})
                per[ph] = {key: curve.get(key, {"status": "not_saved_in_source"}) for key in ("50", "100", "200", "400", "best")}
                if parts[ph]["cost"].get("boosting_rounds_fitted", 0) < 400:
                    per[ph]["400"] = {"status": "not_available", "reason": "model_fitted_fewer_than_400_rounds"}
            e_report = e_phase_report(p, y, fam, n_bootstrap=args.n_bootstrap)
            identity = source_name if source_name in {"V20", "V50"} else run_id
            report["runs"][identity] = dict(source_experiment=run_id, model_seed=42, feature_view="v4",
                ul_raw_by_phase=ll, ul_raw=float(np.mean(list(ll.values()))),
                calibration_in_the_large=calibration, subgroup_log_loss_user=subgroup,
                calibration_E_by_user=e_by_user, E_false_positives=fp, e_class_raw=e_report,
                source_cost=parts["P1"]["cost"], round_diagnostics=per,
                teacher_subgroup_status="pending_data_team_mapping",
                probability_basis="raw_normalized_after_float32_storage",
                prior_correction="identity_for_natural_FullTrain_same_enrollment_prior",
                original_prediction_sha256={ph: parts[ph]["prediction_sha256"] for ph in PHASES})
            private[identity] = (p, y, fam)
    v50, v20 = private["V50"], private["V20"]
    report["V50_minus_V20_CI"] = paired_family_logloss_ci(v50[0], v20[0], v50[1], v50[2], n_bootstrap=args.n_bootstrap)
    courses = {ph: dev[f"dev_phase{ph[-1]}"].course_id.to_numpy() for ph in PHASES}
    loss50 = {ph: row_log_loss(v50[0][ph], v50[1][ph]) for ph in PHASES}
    loss20 = {ph: row_log_loss(v20[0][ph], v20[1][ph]) for ph in PHASES}
    report["V50_minus_V20_course_contributions"] = course_contributions(
        np.concatenate(list(loss50.values())), np.concatenate(list(loss20.values())),
        np.concatenate([courses[ph] for ph in PHASES]))
    report["V50_minus_V20_course_contributions_by_phase"] = {
        ph: course_contributions(loss50[ph], loss20[ph], courses[ph]) for ph in PHASES}
    report["V50_and_fullB_with_predictions_equal"] = all(
        np.array_equal(private["V50"][0][ph], private["full_B_with"][0][ph]) for ph in PHASES)
    args.output.mkdir(parents=True)
    write_json(args.output / "V50_V20_E_diagnostics.json", report)
    md = ["# Phân tích bổ sung ngày 25 tháng 10 năm 2026", "",
          "Tính lại từ dự đoán dev đã lưu; không train mới, không đọc test. Các lượt là V4 view, seed42, chưa phải CD3.",
          "", "## V50 theo phase và người học", "",
          table(["Phase", "LL", "E FP", "E FP người học mới", "Tỷ lệ người học mới trong FP E"],
                [[ph, report['runs']['V50']['ul_raw_by_phase'][ph],
                  *[report['runs']['V50']['E_false_positives'][ph][k] for k in
                    ('false_positive_E','new_user_false_positive_E','new_user_fraction_in_false_positive_E')]] for ph in PHASES]),
          "", "## Chỉ số E gộp bốn phase và CI theo họ", "",
          "Mỗi lượt ghi danh xuất hiện bốn lần trong phần gộp; số họ không nhân bốn. CI 2.000 lượt lấy mẫu họ cùng mọi phase. Không coi các phase là replicate độc lập.", ""]
    rows = []
    for name, run in report['runs'].items():
        e = run['e_class_raw']['pooled_four_phases']
        rows.append([name, e['nll_E'], e['auc_E'], e['ap_E'], e['recall_E'], e['precision_E'], e['n_families_with_E'], e['n_families_with_TP_E']])
    md += [table(["Lượt", "NLL E", "AUC E", "AP E", "Recall E", "Precision E", "Họ có E", "Họ có E đúng"], rows),
           "", "CI95 NLL/AUC/AP và toàn bộ theo phase ở JSON kèm theo.",
           "", "## Gain và chênh lệch theo khóa", "",
           f"Tỷ trọng total gain của năm mã ở V50: {report['runs']['V50']['source_cost'].get('v4_code_gain_share'):.6f}.",
           "", table(["Course ID", "Đóng góp vào LL V50−V20"], report['V50_minus_V20_course_contributions']['top'].items()),
           "", "## Phạm vi và phần chờ", "",
           "Đã có calibration theo phase/lớp, subgroup người học, các mốc 50/100/200/best được lưu trong artifact; mốc 400 không có vì V50 fit 351 round. Mapping teacher_subgroup còn chờ nhóm dữ liệu. Các hash dự đoán gốc và chi phí từng lượt nằm trong JSON.",
           "Không có learner CD3, margin/rho được duyệt hay panel/test mới trong báo cáo này."]
    (args.output / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"DONE {args.output}; fits=0; test=false", flush=True)


if __name__ == "__main__":
    main()
