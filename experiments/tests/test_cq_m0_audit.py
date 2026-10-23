"""Kiểm các hàm lõi của audit M0, không đọc dữ liệu CQ riêng tư."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_cq_v1_m0 import _hash_rows, _quality_class
from scripts.run_cq_v1_m0_probes import _cutoff_probe, _missingness_probe


def test_quality_class_keeps_registered_boundaries() -> None:
    values = pd.Series([0.0, 0.1999, 0.2, 0.3999, 0.4, 1.0])
    assert _quality_class(values).tolist() == ["E", "E", "G", "G", "A", "A"]


def test_public_split_hash_is_order_invariant() -> None:
    frame = pd.DataFrame({"course_id": [2, 1], "split": ["test", "train"]})
    assert _hash_rows(frame) == _hash_rows(frame.iloc[::-1])


def test_v3_missingness_and_cutoff_probes_use_dev_only(tmp_path) -> None:
    source = tmp_path / "source"
    numeric = ["f0", "f1"]
    for phase_index in range(1, 5):
        phase = f"P{phase_index}"
        raw = source / "supervised_raw" / phase
        raw.mkdir(parents=True)
        for split, size in (("train", 30), ("dev", 12)):
            labels = np.resize(np.array(["E", "G", "A"]), size)
            frame = pd.DataFrame({
                "y_alpha_1": labels,
                "f0": np.where(labels == "E", np.nan, 1.0),
                "f1": np.where(labels == "A", np.nan, 2.0),
            })
            frame.to_parquet(raw / f"{split}.parquet", index=False)
        audit = source / "audit" / phase
        audit.mkdir(parents=True)
        timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
        pd.DataFrame({
            "user_id": [f"u{i}" for i in range(10)],
            "course_id": [f"c{i}" for i in range(10)],
            "enroll_time": timestamps,
            "cutoff_time": timestamps + pd.Timedelta(days=2),
            "min_feature_event_time": timestamps + pd.Timedelta(hours=1),
            "max_feature_event_time": timestamps + pd.Timedelta(days=1),
            "n_out_of_window_events": np.zeros(10, dtype=int),
        }).to_parquet(audit / "part-000.parquet", index=False)

    missingness = _missingness_probe(source, "y_alpha_1", numeric, seed=11)
    cutoff = _cutoff_probe(source, tmp_path / "private.csv", seed=11)
    assert missingness["status"] == "PASS_PROBE_COMPLETED"
    assert missingness["missing_mask_used_as_production_feature"] is False
    assert cutoff["status"] == "PASS_20_PRIVATE_CASES"
    assert cutoff["sample_size"] == 20
    assert (tmp_path / "private.csv").is_file()
