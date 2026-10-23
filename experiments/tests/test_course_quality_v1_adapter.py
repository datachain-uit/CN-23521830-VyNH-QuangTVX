"""Kiểm tra adapter CQ v1 giữ Pool B, bốn dev phase và khóa test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from skq_exp.data.adapters import prepare_external_dataset
from skq_exp.data.prepare import load_processed


def test_course_quality_v1_pool_b_and_four_dev_phases(tmp_path):
    source = tmp_path / "phase_features_v2"
    processed = tmp_path / "processed"
    (source / "model_ready").mkdir(parents=True)
    feature_columns = [f"f{i}" for i in range(56)] + ["phase_fraction"]
    metadata = {
        "distance_feature_columns": feature_columns,
        "categorical_features": ["learner_code"],
        "model_feature_columns": [*feature_columns, "learner_code"],
        "target_columns": ["q_alpha_0p1", "y_alpha_0p1", "q_alpha_1", "y_alpha_1"],
        "preprocessing_scope": "train-only per phase",
    }
    (source / "feature_columns.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "validation.json").write_text(json.dumps({"all_good": True}), encoding="utf-8")
    for name in ("_SUCCESS.json", "preprocessing.json", "run_manifest.json"):
        (source / name).write_text("{}", encoding="utf-8")

    for phase_index in range(1, 5):
        phase = f"P{phase_index}"
        phase_dir = source / "model_ready" / phase
        phase_dir.mkdir()
        for split, size, course in (("train", 6, "train"), ("dev", 3, "dev"), ("test", 3, "test")):
            frame = pd.DataFrame({column: np.arange(size, dtype=float) for column in feature_columns})
            frame["user_id"] = [f"u{i}" for i in range(size)]
            frame["course_id"] = course
            frame["phase"] = phase
            frame["split"] = split
            frame["learner_code"] = np.arange(size, dtype=float) % 2
            frame["y_alpha_1"] = (["E", "G", "A"] * 2)[:size]
            frame["q_alpha_1"] = np.linspace(0.1, 0.5, size)
            frame["y_alpha_0p1"] = frame["y_alpha_1"]
            frame["q_alpha_0p1"] = frame["q_alpha_1"]
            frame.to_parquet(phase_dir / f"{split}.parquet", index=False)

    manifest = prepare_external_dataset(
        "course_quality_v1", source, processed,
        validate_expected_rows=False,
    )
    data = load_processed("course_quality_v1", processed, mmap=False)
    assert data["X_train"].shape == (24, 58)
    assert data["X_selector_train"].shape == (24, 56)
    assert data["phase_train"].tolist() == [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6
    assert all(data[f"X_dev_phase{phase}"].shape == (3, 58) for phase in range(1, 5))
    assert np.array_equal(data["X_dev"], data["X_dev_phase4"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["test_locked"] is True
    assert payload["confirmatory_eligible"] is False
    contract = json.loads((processed / "course_quality_v1" / "feature_contract.json").read_text(encoding="utf-8"))
    assert "learner_code" in contract["learner_columns"]
    assert "learner_code" not in contract["selector_columns"]


def test_course_quality_v3_dev_bundle_refits_one_pool_b_preprocessor(tmp_path):
    """V3 raw train/dev works without test and keeps phase signal after scaling."""
    source = tmp_path / "phase_features_v3" / "course_random_split"
    processed = tmp_path / "processed"
    (source / "supervised_raw").mkdir(parents=True)
    numeric_columns = [f"f{i}" for i in range(56)] + ["phase_fraction"]
    metadata = {
        "version": "phase-features-v3-test",
        "raw_numeric_features": numeric_columns,
        "distance_feature_columns": numeric_columns,
        "categorical_features": ["learner_code"],
        "model_feature_columns": [*numeric_columns, "learner_code"],
        "target_columns": ["q_alpha_0p1", "y_alpha_0p1", "q_alpha_1", "y_alpha_1"],
        "preprocessing_scope": "per-phase source preprocessing",
    }
    (source / "feature_columns.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "validation.json").write_text(json.dumps({"all_good": True}), encoding="utf-8")
    for name in ("_SUCCESS.json", "preprocessing.json", "run_manifest.json"):
        (source / name).write_text("{}", encoding="utf-8")

    for phase_index, fraction in enumerate((0.25, 0.50, 0.75, 0.90), start=1):
        phase = f"P{phase_index}"
        phase_dir = source / "supervised_raw" / phase
        phase_dir.mkdir()
        for split, size, course in (("train", 3, f"train-{phase}"), ("dev", 2, f"dev-{phase}")):
            frame = pd.DataFrame({column: np.arange(size, dtype=float) for column in numeric_columns})
            frame.loc[0, "f0"] = np.nan
            frame["phase_fraction"] = fraction
            frame["user_id"] = [f"{split}-u{i}" for i in range(size)]
            frame["course_id"] = course
            frame["phase"] = phase
            frame["split"] = split
            frame["learner_code"] = np.arange(size, dtype=float) % 2
            frame["y_alpha_1"] = (["E", "G", "A"] * 2)[:size]
            frame["q_alpha_1"] = np.linspace(0.1, 0.5, size)
            frame["y_alpha_0p1"] = frame["y_alpha_1"]
            frame["q_alpha_0p1"] = frame["q_alpha_1"]
            frame.to_parquet(phase_dir / f"{split}.parquet", index=False)

    manifest = prepare_external_dataset(
        "course_quality_v1", source, processed,
        validate_expected_rows=False,
    )
    data = load_processed("course_quality_v1", processed, mmap=False)
    assert data["X_train"].shape == (12, 58)
    phase_feature = data["X_train"][:, 56]
    assert len(np.unique(np.round(phase_feature, decimals=6))) == 4
    assert not np.allclose(phase_feature, 0.0)
    assert "X_test" not in data
    assert not (processed / "course_quality_v1" / "X_test.npy").exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["test_locked"] is True
    assert payload["test_data_packaged"] is False
    assert payload["fit_scope"] == "median/mean/std fitted once on concatenated P1-P4 train"
