from __future__ import annotations

import json

import numpy as np
import pandas as pd

from skq_exp.data.adapters import get_dataset_adapter, prepare_external_dataset
from skq_exp.data.prepare import load_processed


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "chapter": ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2"],
        "score": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "teacher_id": ["T1", "T1", "T2", "T2", "T3", "T3"],
        "school_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "Rank": ["1", "1", "2", "2", "3", "3"],
        "type_of_rank": ["A", "A", "B", "B", "C", "C"],
        "label": ["leak"] * 6,
        "user_id_enc": [1, 1, 2, 2, 3, 3],
        "course_id_enc": [10, 10, 20, 20, 30, 30],
        "euclid_to_111": [0.1] * 6,
        "label_f": ["excellent", "good", "average", "good", "excellent", "average"],
    })


def _test_frame() -> pd.DataFrame:
    frame = _train_frame().iloc[:3].copy()
    chapter = frame.pop("chapter").str.split(".", expand=True)
    frame["chapter_1"] = chapter[0].astype(int)
    frame["chapter_2"] = chapter[1].astype(int)
    frame["chapter_3"] = 0
    frame["phase"] = [1, 2, 3]
    return frame


def test_prepare_course_quality_preserves_four_snapshots(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    train = _train_frame()
    train.to_csv(raw / "train_med.csv", index=False)
    train.iloc[:3].to_csv(raw / "val_med.csv", index=False)
    for phase in range(1, 5):
        test = _test_frame()
        test["score"] = test["score"].fillna(0) + phase
        test.to_csv(raw / f"test_med_{phase}.csv", index=False)

    assert get_dataset_adapter("course_quality_med_v1").dataset_id == (
        "course_quality_med_v1"
    )
    manifest = prepare_external_dataset(
        "course_quality_med_v1", raw, processed, validate_expected_rows=False,
    )
    assert manifest.exists()
    data = load_processed("course_quality_med_v1", processed, mmap=False)
    assert data["X_train"].shape == (6, 8)
    assert data["X_dev"].shape == (3, 8)
    assert np.array_equal(data["X_test"], data["X_test_phase4"])
    assert set(np.unique(data["y_train"])) == {0, 1, 2}
    assert data["X_train"][2, 0] == 4.0
    schema = json.loads((manifest.parent / "schema.json").read_text(encoding="utf-8"))
    assert "label" not in schema["processed_columns"]
    assert "user_id_enc" not in schema["processed_columns"]
    assert schema["preprocessing_contract"].endswith("_v1")
