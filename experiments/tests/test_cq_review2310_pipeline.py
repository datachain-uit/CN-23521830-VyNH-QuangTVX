"""CPU toy integration of raw CD3 preparation, runner and required Y25 evidence.

No real CQ inputs or test split are read. These checks are infrastructure
evidence only and must not be reported as model quality on CQ.
"""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_cq_cd3_dev import prepare
from scripts.compute_cq_v4_dev_endpoints import compute
from scripts.compare_cq_review2310 import compare
from skq_exp.artifacts import atomic_json, sha256_file, ArtifactLayout, ArtifactStorageError
from skq_exp.config import ExperimentConfig
from skq_exp.data.prepare import load_processed
from skq_exp.experiments.runner import run_config
from skq_exp.experiments.runner import _save_evaluation
from skq_exp.experiments.cq_review_1023 import xgb_round_diagnostics
from skq_exp.methods.result import SelectionResult
from skq_exp.training.evaluator import fit_selection_learner
from test_cq_review2310_runtime import _frames


def _hashes(root):
    rows = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}"
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sources(tmp_path):
    raw_root, v4 = tmp_path / "raw", tmp_path / "v4"
    raw_root.mkdir()
    v4.mkdir()
    train, dev, full = _frames()
    paths = {"pool_b/train.parquet": train, "fulltrain_a/train.parquet": full,
             **{f"pool_b/dev_phase{p[-1]}.parquet": f for p, f in dev.items()}}
    for relative, frame in paths.items():
        for root in (raw_root / "raw_features", v4):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target, index=False)
    atomic_json(v4 / "splits.json", {"group_unit": "course_family_id", "seed": 42})
    atomic_json(v4 / "run_manifest.json", {"dataset_version": "CQ V4", "fixture": True})
    pd.DataFrame({"course_family_id": ["train-family", "dev-family"]}).to_parquet(v4 / "family_splits.parquet")
    pd.DataFrame({"course_id": ["train-course", "dev-course"]}).to_parquet(v4 / "course_splits.parquet")
    _hashes(raw_root)
    _hashes(v4)
    return raw_root, v4


def test_CD3_prepare_LR_runner_parquet_and_LL_report_on_toy(tmp_path):
    raw, v4 = _sources(tmp_path)
    bundle, processed = tmp_path / "cd3_bundle", tmp_path / "processed"
    prepare(raw, v4, bundle, processed)
    data = load_processed("course_quality_v4", processed)
    assert data["feature_contract"]["feature_view"] == "cd3"
    assert not any(k.startswith("X_test") for k in data)
    config = ExperimentConfig.from_json(Path(__file__).parents[1] / "configs/cq_v4_review2310_lr_01_v1.json")
    config = replace(config, experiment_id="toy23", protocol_id="p23", paths=replace(
        config.paths, processed_root=processed, artifact_root=tmp_path / "artifacts"),
        learner_params={"lr": {"standardize": True, "max_iter": 200, "C": 1., "solver": "lbfgs"}})
    rows = run_config(config)
    assert len(rows) == 4 and all(r["status"] == "success" for r in rows)
    outputs = list((tmp_path / "artifacts").rglob("predictions.parquet"))
    assert len(outputs) == 4
    frame = pd.read_parquet(outputs[0])
    assert {"row_id", "phase", "p_E", "p_G", "p_A", "run_id", "y"} <= set(frame)
    assert all(frame[c].dtype == np.float32 for c in ("p_E", "p_G", "p_A"))
    assert frame.row_id.astype(str).str.startswith("dev-e").all()
    assert set(frame.phase) <= {"P1", "P2", "P3", "P4"}
    result = compute(bundle, tmp_path / "artifacts", review_1023=True)
    report = result["runs"][0]
    assert report["primary_endpoint"] == "mean_log_loss_P1_P4"
    assert np.isfinite(report["ul_log_loss"]["ul"])
    assert "ul_log_loss" in report["a8_prior_correction"]
    assert set(report["per_label_by_phase"]) == {"P1", "P2", "P3", "P4"}
    assert report["fit_diagnostics_by_phase"]["P1"]["lr_gender_encoding"]["fit_scope"] == "all_pool_b_train"
    assert "teacher" in report["subgroup_log_loss"]["P1"]
    paired = compare(bundle, tmp_path / "artifacts", "toy23", "toy23", n_bootstrap=100)
    assert paired["paired_family_ci"]["ci95"] == [0., 0.]
    assert paired["course_contributions_mean_phases"]["total"] == 0.
    with pytest.raises(FileExistsError):
        prepare(raw, v4, bundle, processed)


def test_preparation_rejects_changed_V4_q_before_fitting(tmp_path):
    raw, v4 = _sources(tmp_path)
    path = raw / "raw_features/pool_b/train.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "q_alpha_1"] += .001
    frame.to_parquet(path, index=False)
    _hashes(raw)
    with pytest.raises(ValueError, match="immutable V4 spine"):
        prepare(raw, v4, tmp_path / "out", tmp_path / "processed")


def test_actual_CPU_XGB_records_rounds_and_trees_without_extrapolating():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(45, 4)).astype(np.float32), np.tile(np.arange(3), 15)
    selection = SelectionResult(indices=np.arange(45), weights=np.ones(45), requested_rows=45,
                                realized_rows=45, budget_mode="exact_total", method_id="toy")
    fitted = fit_selection_learner(selection, X, y, learner_id="xgb", max_threads=1,
        class_weight_policy="natural", learner_params={"n_estimators": 6, "max_depth": 1, "device": "cpu"},
        X_inner_val=X, y_inner_val=y,
        early_stopping={"enabled": True, "source": "dev", "patience": 2, "phases": ["P4"]})
    cost = fitted.base_cost
    assert cost["trees_per_boosting_round"] == 3
    assert cost["trees_used_for_prediction"] == 3*(cost["best_iteration"]+1)
    assert cost["trees_fitted"] == 3*cost["boosting_rounds_fitted"]
    curve = xgb_round_diagnostics(fitted.model, X, y, [1, 2, 400])
    assert curve["points"]["400"]["status"] == "not_available"
    assert curve["points"]["best"]["status"] == "success"


def test_required_parquet_storage_limit_is_failure_not_silent_success(tmp_path):
    run = tmp_path / "r"
    atomic_json(run / "manifest.json", {"run_id": "toy"})
    pred = {"y_pred": np.arange(3), "y_true": np.arange(3),
            "y_prob": np.eye(3), "row_id": np.arange(3), "source_row_id": np.array(["a", "b", "c"]),
            "phase": np.repeat("P1", 3)}
    with pytest.raises(ArtifactStorageError, match="Y25"):
        _save_evaluation(run, None, {}, pred, policy={
            "save_predictions": True, "save_y_true": True, "save_probabilities": True,
            "save_predictions_parquet": True, "max_prediction_mb_per_run": 1e-12,
        }, layout=ArtifactLayout(tmp_path / "a"), experiment_id="toy")
