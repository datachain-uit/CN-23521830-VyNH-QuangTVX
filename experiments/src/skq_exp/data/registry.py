"""File này ghi danh tính và cách chia của mọi dataset đã đăng ký.

Tên dataset ID không phụ thuộc tên file tải về. Các con số dự kiến giúp phát hiện
nhầm phiên bản trước khi một dataset đi vào thực nghiệm. Cách đọc dữ liệu ngoài
nằm trong ``data/adapters``; registry này chỉ giữ metadata khoa học ổn định.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    title: str
    provider: str
    source: str
    expected_rows: int
    expected_features: int
    expected_classes: int
    split_policy: str


DATASET_SPECS = {
    spec.dataset_id: spec for spec in [
        DatasetSpec(
            "adult_uci2_v1", "Adult", "UCI",
            "https://archive.ics.uci.edu/dataset/2/adult",
            48_842, 14, 2, "canonical_test_train_only_dev15",
        ),
        DatasetSpec(
            "creditcard_ulb2013_v1", "Credit Card Fraud", "Kaggle",
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            284_807, 30, 2, "duplicate_group_safe_70_15_15",
        ),
        DatasetSpec(
            "letter_uci59_v1", "Letter Recognition", "UCI",
            "https://archive.ics.uci.edu/dataset/59/letter+recognition",
            20_000, 16, 26, "canonical_16k_4k_train_only_dev15",
        ),
        DatasetSpec(
            "covertype_uci31_v1", "Covertype", "UCI",
            "https://archive.ics.uci.edu/dataset/31/covertype",
            581_012, 54, 7, "duplicate_group_safe_70_15_15",
        ),
        DatasetSpec(
            "jannis_openml41168_v1", "Jannis", "OpenML",
            "https://www.openml.org/d/41168",
            83_733, 54, 4, "stratified_70_15_15",
        ),
        DatasetSpec(
            "helena_openml41169_v1", "Helena", "OpenML",
            "https://www.openml.org/d/41169",
            65_196, 27, 100, "stratified_70_15_15_all_classes",
        ),
        DatasetSpec(
            "course_quality_med_v1", "CourseQuality MED", "User-provided Kaggle snapshot",
            "https://www.kaggle.com/datasets/hoangzyyng/cq-med",
            3_297_128, 59, 3,
            "provided_train_val_four_temporal_tests_test_locked",
        ),
        DatasetSpec(
            "course_quality_v1", "CourseQuality enrollment-phase v1 / data V3", "Private group snapshot",
            "private://phase_features_v3/course_random_split",
            178_598, 56, 3,
            "provided_course_random_disjoint_train_dev_test_pool_b_four_phases_test_locked",
        ),
        DatasetSpec(
            "course_quality_v1_afelo0p1", "CourseQuality enrollment-phase v1 / data V3 alpha=0.1",
            "Private group snapshot", "private://phase_features_v3/course_random_split",
            178_598, 56, 3,
            "provided_course_random_disjoint_train_dev_test_pool_b_four_phases_test_locked_sensitivity",
        ),
        DatasetSpec(
            "course_quality_v4", "CourseQuality family-temporal V4", "Private group snapshot",
            "private://course_quality_v4/family_temporal_dev_only",
            0, 56, 3,
            "course_family_latest20_test_train70_dev10_seed42_test_locked",
        ),
    ]
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_id]
    except KeyError as error:
        raise KeyError(f"Dataset chưa đăng ký: {dataset_id}") from error
