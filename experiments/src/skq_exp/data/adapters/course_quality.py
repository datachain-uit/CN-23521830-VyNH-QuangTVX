"""Adapter CourseQuality MED trên khung CSV pre-split dùng chung.

Điểm riêng của dataset này chỉ còn layout sáu file, nhãn, cột cần loại và cách
tách ``chapter``. Luồng lưu artifact/audit/hash nằm ở base adapter nên dataset mới
không cần sao chép toàn bộ code CourseQuality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .presplit_csv import PresplitCSVLayout, TrainOnlyOrdinalCSVAdapter


class CourseQualityMEDAdapter(TrainOnlyOrdinalCSVAdapter):
    dataset_id = "course_quality_med_v1"
    layout = PresplitCSVLayout(
        train_file="train_med.csv",
        dev_file="val_med.csv",
        test_files=tuple(f"test_med_{phase}.csv" for phase in range(1, 5)),
        canonical_test_index=-1,
    )
    expected_rows_by_file = {
        "train_med.csv": 2_637_700,
        "val_med.csv": 329_713,
        "test_med_1.csv": 329_715,
        "test_med_2.csv": 329_715,
        "test_med_3.csv": 329_715,
        "test_med_4.csv": 329_715,
    }
    expected_processed_features = 59
    group_columns = ("user_id_enc", "course_id_enc")
    target_column = "label_f"
    label_mapping = {"excellent": 0, "good": 1, "average": 2}
    drop_columns = frozenset({
        "label", "user_id", "course_id", "user_id_enc", "course_id_enc",
        "euclid_to_111", "phase",
    })
    preprocessing_contract = "course_quality_author_methodology_train_only_fit_v1"
    evidence_role = "integration_debug_until_group_disjoint_gate_passes"
    fit_scope = "categorical maps and numeric medians fitted on train only"

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "chapter" not in frame.columns:
            return frame
        parts = frame.pop("chapter").astype(str).str.split(".", expand=True)
        for level in range(3):
            values = (
                parts[level]
                if level < parts.shape[1]
                else pd.Series(0, index=frame.index, dtype=np.int64)
            )
            frame[f"chapter_{level + 1}"] = (
                pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64)
            )
        return frame
