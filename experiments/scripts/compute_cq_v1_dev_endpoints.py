"""Tính UL/EC dev từ prediction artifacts CQ v1, không truy cập test.

Prediction cấp dòng được ghép với Parquet dev theo đúng thứ tự adapter và chỉ
ghi dưới ``artifacts/privacy`` (đã ignore). Báo cáo tổng hợp không chứa ID.
``course_id_equals_root`` chỉ được dùng sau khi nhóm dữ liệu xác nhận bằng văn
bản; chế độ provisional tạo evidence phát triển nhưng không mở confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skq_exp.training.course_metrics import (
    class_conditional_means,
    class_prevalence,
    cq_endpoints,
    prior_correct_predictions,
)


PHASE_BY_EVAL = {f"dev_phase{index}": f"P{index}" for index in range(1, 5)}
LABEL_TO_INT = {"E": 0, "G": 1, "A": 2}
INT_TO_LABEL = np.asarray(["E", "G", "A"], dtype=object)


def _experiment_root(artifact_root: Path, experiment_id: str) -> Path:
    """Trả đúng canonical run root mà experiment runner đang ghi."""
    return artifact_root / "experiments" / experiment_id


def _opaque(values: pd.Series, namespace: str) -> np.ndarray:
    return np.asarray([
        hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()[:24]
        for value in values.astype(str)
    ], dtype=object)


def _common_key(manifest: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(manifest["dataset_id"]), str(manifest["method_id"]),
        int(manifest["selector_seed"]), str(manifest["learner_id"]),
        int(manifest["model_seed"]),
    )


def _source_prior(
    items: list[tuple[Path, dict[str, Any]]],
    pool_prior: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Đọc effective prior; hỗ trợ artifact M0 cũ của FullTrain-B có policy."""
    policies = set()
    recorded = []
    for run_dir, _manifest in items:
        cost_path = run_dir / "cost.json"
        if not cost_path.is_file():
            raise FileNotFoundError(f"Thiếu cost artifact cho A8: {cost_path}")
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        policies.add(cost.get("class_weight_policy"))
        payload = cost.get("effective_class_prior")
        if payload is not None:
            labels = payload.get("labels", [])
            probabilities = payload.get("probabilities", [])
            if len(labels) != len(probabilities):
                raise ValueError("effective_class_prior không thẳng hàng")
            mapped = {
                INT_TO_LABEL[int(label)] if str(label).isdigit() else str(label): float(value)
                for label, value in zip(labels, probabilities)
            }
            recorded.append(mapped)
    if recorded:
        first = recorded[0]
        if any(
            not np.allclose(
                [row[label] for label in INT_TO_LABEL],
                [first[label] for label in INT_TO_LABEL], atol=1e-10,
            )
            for row in recorded[1:]
        ):
            raise ValueError("Effective source prior thay đổi giữa các phase eval")
        return first, "effective_sample_weight_artifact"
    if len(policies) != 1:
        raise ValueError(f"Class-weight policy không nhất quán: {sorted(map(str, policies))}")
    policy = next(iter(policies))
    if policy == "natural":
        return pool_prior, "legacy_fulltrain_natural_reconstructed"
    if policy == "balanced":
        return {label: 1.0 / len(INT_TO_LABEL) for label in INT_TO_LABEL}, "legacy_fulltrain_balanced_reconstructed"
    raise ValueError("Artifact cũ không đủ effective prior để chạy A8")


def compute(
    source: Path,
    artifact_root: Path,
    experiment_id: str,
    *,
    n_min: int,
    root_policy: str,
    target_q: str,
    target_y: str,
) -> dict[str, Any]:
    experiment = _experiment_root(artifact_root, experiment_id)
    manifests = []
    for path in experiment.rglob("manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluation_split") in PHASE_BY_EVAL and (path.parent / "predictions.npz").is_file():
            manifests.append((path.parent, payload))
    if not manifests:
        raise FileNotFoundError(f"Không tìm thấy dev predictions trong {experiment}")

    train_by_phase = {}
    for phase in ("P1", "P2", "P3", "P4"):
        train_by_phase[phase] = pd.read_parquet(
            source / "model_ready" / phase / "train.parquet",
            columns=[target_q, target_y],
        ).rename(columns={target_q: "q", target_y: "y"})
    mu = class_conditional_means(train_by_phase["P1"])
    target_prior_by_phase = {
        phase: class_prevalence(frame["y"])
        for phase, frame in train_by_phase.items()
    }
    pool_prior = class_prevalence(pd.concat(
        [frame["y"] for frame in train_by_phase.values()], ignore_index=True,
    ))
    grouped: dict[tuple[str, str, int, str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for item in manifests:
        grouped.setdefault(_common_key(item[1]), []).append(item)

    private_root = artifact_root / "privacy" / experiment_id
    private_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for key, items in sorted(grouped.items()):
        by_eval = {manifest["evaluation_split"]: (run_dir, manifest) for run_dir, manifest in items}
        missing = sorted(set(PHASE_BY_EVAL) - set(by_eval))
        if missing:
            summaries.append({"identity": list(key), "status": "incomplete", "missing": missing})
            continue
        endpoint_run_id = hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:20]
        frames = []
        for eval_name, phase in PHASE_BY_EVAL.items():
            run_dir, manifest = by_eval[eval_name]
            prediction = np.load(run_dir / "predictions.npz")
            if "y_prob" not in prediction:
                raise ValueError(f"{run_dir} thiếu y_prob; config phải save_probabilities=true")
            source_frame = pd.read_parquet(
                source / "model_ready" / phase / "dev.parquet",
                columns=["user_id", "course_id", target_q, target_y],
            )
            y_true_text = source_frame[target_y].astype(str).to_numpy(dtype=object)
            if not set(np.unique(y_true_text)) <= set(LABEL_TO_INT):
                raise ValueError(f"Nhãn ngoài E/G/A ở {eval_name}")
            y_true_numeric = np.asarray([LABEL_TO_INT[label] for label in y_true_text], dtype=int)
            if len(source_frame) != len(prediction["y_pred"]):
                raise ValueError(f"Lệch số dòng {eval_name}: source={len(source_frame)}, prediction={len(prediction['y_pred'])}")
            if "y_true" in prediction and not np.array_equal(y_true_numeric, prediction["y_true"].astype(int)):
                raise ValueError(f"Thứ tự y_true không khớp source ở {eval_name}")
            probabilities = np.asarray(prediction["y_prob"], dtype=float)
            if probabilities.shape != (len(source_frame), 3):
                raise ValueError(f"Probability shape sai ở {eval_name}: {probabilities.shape}")
            raw_prediction = np.asarray(prediction["y_pred"])
            if not set(np.unique(raw_prediction.astype(int))) <= {0, 1, 2}:
                raise ValueError(f"y_pred ngoài 0/1/2 ở {eval_name}")
            course = _opaque(source_frame["course_id"], "course")
            learner = _opaque(source_frame["user_id"], "learner")
            frames.append(pd.DataFrame({
                "run_id": endpoint_run_id,
                "course_id": course,
                "course_root_id": course,
                "learner_id": learner,
                "phase": phase,
                "q_true": source_frame[target_q].to_numpy(dtype=float),
                "y_true": y_true_text,
                "y_pred": INT_TO_LABEL[raw_prediction.astype(int)],
                "p_E": probabilities[:, 0], "p_G": probabilities[:, 1], "p_A": probabilities[:, 2],
            }))
        private = pd.concat(frames, ignore_index=True)
        private_path = private_root / f"{endpoint_run_id}.parquet"
        private.to_parquet(private_path, index=False)
        pcc = cq_endpoints(private, mu, n_min=n_min, mode="pcc")
        cc = cq_endpoints(private, mu, n_min=n_min, mode="cc")
        source_prior, source_prior_evidence = _source_prior(items, pool_prior)
        corrected_private = prior_correct_predictions(
            private, source_prior, target_prior_by_phase,
        )
        corrected_pcc = cq_endpoints(corrected_private, mu, n_min=n_min, mode="pcc")
        corrected_cc = cq_endpoints(corrected_private, mu, n_min=n_min, mode="cc")
        summaries.append({
            "status": "success",
            "dataset_id": key[0], "method_id": key[1], "selector_seed": key[2],
            "learner_id": key[3], "model_seed": key[4],
            "endpoint_run_id": endpoint_run_id,
            "evaluation_level": ["enrollment", "course"],
            "aggregation_primary": "PCC",
            "root_policy": root_policy,
            "confirmatory_eligible": root_policy == "course_id_equals_root",
            "n_min": n_min,
            "mu_train": mu,
            "pcc": pcc,
            "cc_sensitivity": cc,
            "a8_prior_correction": {
                "status": "success",
                "retrained": False,
                "assumption": "label_shift_only",
                "source_prior": source_prior,
                "source_prior_evidence": source_prior_evidence,
                "target_prior_by_phase": target_prior_by_phase,
                "pcc": corrected_pcc,
                "cc_sensitivity": corrected_cc,
                "delta_ul_corrected_minus_original": (
                    corrected_pcc["ul_mean_p1_p4"] - pcc["ul_mean_p1_p4"]
                ),
                "delta_ec_original_minus_corrected": (
                    None if pcc["ec_mean_p1_p4"] is None or corrected_pcc["ec_mean_p1_p4"] is None
                    else pcc["ec_mean_p1_p4"] - corrected_pcc["ec_mean_p1_p4"]
                ),
            },
            "private_predictions": str(private_path),
        })
    output = experiment / "cq_dev_endpoints.json"
    output.write_text(json.dumps({
        "schema_version": 1,
        "experiment_id": experiment_id,
        "split": "dev_only",
        "test_predictions_read": False,
        "root_policy": root_policy,
        "confirmatory_eligible": root_policy == "course_id_equals_root",
        "runs": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "runs": len(summaries), "successful": sum(row.get("status") == "success" for row in summaries)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--root-policy", choices=("provisional_course_id", "course_id_equals_root"), default="provisional_course_id")
    parser.add_argument("--target-q", default="q_alpha_1")
    parser.add_argument("--target-y", default="y_alpha_1")
    args = parser.parse_args()
    result = compute(
        Path(args.input_dir).resolve(), Path(args.artifact_root).resolve(), args.experiment_id,
        n_min=args.n_min, root_policy=args.root_policy, target_q=args.target_q, target_y=args.target_y,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
