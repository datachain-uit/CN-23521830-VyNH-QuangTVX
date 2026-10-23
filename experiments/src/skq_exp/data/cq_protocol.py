"""Contract dữ liệu CQ v1 ở cấp lượt ghi danh theo Review 12–13/10/2026.

Module này chỉ hiện thực các quyết định đã xác nhận: q là trung bình LOQV theo
chương, ngưỡng E/G/A là 0.2/0.4 (nhỏ là tốt), P1–P4 là 25/50/75/90%, split
temporal theo course root và Pool B có bốn snapshot mỗi enrollment. Ngưỡng độ tin
cậy timestamp chưa được review chốt nên bắt buộc truyền vào, không có mặc định.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


CLASSES = ("E", "G", "A")
CUTS = (0.2, 0.4)
PHASES = {"P1": 0.25, "P2": 0.50, "P3": 0.75, "P4": 0.90}


@dataclass(frozen=True)
class EnrollmentBuildAudit:
    source_rows: int
    enrollment_rows: int
    loqv_was_divided_by_sqrt3: bool
    unopened_chapter_policy: str
    minimum_q: float
    maximum_q: float
    chapter_count_min: int
    chapter_count_max: int


@dataclass(frozen=True)
class SplitDecisionThresholds:
    max_timestamp_missing_rate: float
    min_test_courses_per_class: int = 10
    max_dev_ci_half_width: float = 0.005

    def validate(self) -> None:
        if not 0 <= self.max_timestamp_missing_rate <= 1:
            raise ValueError("max_timestamp_missing_rate phải nằm trong [0, 1]")
        if self.min_test_courses_per_class <= 0:
            raise ValueError("min_test_courses_per_class phải dương")
        if self.max_dev_ci_half_width <= 0:
            raise ValueError("max_dev_ci_half_width phải dương")


def quality_class(q: np.ndarray | pd.Series) -> np.ndarray:
    """[0,.2)->E, [.2,.4)->G, [.4,1]->A; giá trị nhỏ là tốt."""
    values = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("q phải hữu hạn và nằm trong [0, 1]")
    positions = np.searchsorted(np.asarray(CUTS), values, side="right")
    return np.asarray(CLASSES, dtype=object)[positions]


def enrollment_table(
    chapter_rows: pd.DataFrame,
    *,
    learner_col: str = "learner_id",
    course_col: str = "course_id",
    loqv_col: str = "euclid_to_111",
    already_normalized: bool | None = None,
    unopened_chapter_policy: Literal["included_as_rows", "excluded_not_open"],
) -> tuple[pd.DataFrame, EnrollmentBuildAudit]:
    """Dựng một dòng (k,c); policy chương chưa mở phải được khai báo rõ."""
    required = {learner_col, course_col, loqv_col}
    missing = sorted(required - set(chapter_rows.columns))
    if missing:
        raise KeyError(f"Thiếu cột dựng enrollment: {missing}")
    if unopened_chapter_policy not in {"included_as_rows", "excluded_not_open"}:
        raise ValueError("Phải khai báo quy tắc chương chưa mở trong data card")
    values = pd.to_numeric(chapter_rows[loqv_col], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("LOQV chứa giá trị thiếu/không phải số")
    divide = bool(values.max(initial=0.0) > 1.0 + 1e-9) if already_normalized is None else not already_normalized
    normalized = values / np.sqrt(3.0) if divide else values
    if np.any(normalized < -1e-9) or np.any(normalized > 1 + 1e-9):
        raise ValueError("LOQV ngoài [0,1] sau quy tắc chuẩn hóa")
    grouped = chapter_rows.assign(_loqv=normalized).groupby(
        [learner_col, course_col], sort=True,
    )["_loqv"]
    table = grouped.agg(q="mean", n_chapters="size").reset_index()
    table["y"] = quality_class(table["q"])
    counts = table["n_chapters"].to_numpy(dtype=int)
    audit = EnrollmentBuildAudit(
        source_rows=int(len(chapter_rows)),
        enrollment_rows=int(len(table)),
        loqv_was_divided_by_sqrt3=divide,
        unopened_chapter_policy=unopened_chapter_policy,
        minimum_q=float(table["q"].min()),
        maximum_q=float(table["q"].max()),
        chapter_count_min=int(counts.min()),
        chapter_count_max=int(counts.max()),
    )
    return table, audit


def validate_enrollment_table(
    table: pd.DataFrame,
    *,
    learner_col: str = "learner_id",
    course_col: str = "course_id",
) -> dict[str, Any]:
    """Kiểm tra D09: cặp duy nhất, y=tau(q), có biến thiên nhãn trong khóa."""
    required = {learner_col, course_col, "q", "y", "n_chapters"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Enrollment table thiếu cột: {missing}")
    if table.duplicated([learner_col, course_col]).any():
        raise ValueError("Mỗi (learner, course) phải xuất hiện đúng một lần")
    expected = quality_class(table["q"])
    if not np.array_equal(expected, table["y"].astype(str).to_numpy()):
        raise ValueError("Nhãn y không khớp tau(q)")
    variation = table.groupby(course_col)["y"].nunique()
    varying_courses = int((variation > 1).sum())
    if varying_courses == 0:
        raise ValueError("Không có within-course label variation; cần audit đơn vị dòng")
    return {
        "schema_version": 1,
        "unique_enrollment_pairs": True,
        "label_formula_matches": True,
        "courses": int(variation.size),
        "courses_with_label_variation": varying_courses,
        "label_counts": table["y"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
    }


def phase_cutoffs(
    enroll_time: pd.Series,
    course_end: pd.Series,
) -> pd.DataFrame:
    """t_kc(p)=e_kc+lambda_p(T_c-e_kc), theo TEMPO-DQ Step 3."""
    enrolled = pd.to_datetime(pd.Series(enroll_time), errors="coerce").reset_index(drop=True)
    ended = pd.to_datetime(pd.Series(course_end), errors="coerce").reset_index(drop=True)
    if enrolled.isna().any() or ended.isna().any():
        raise ValueError("Timestamp enrollment/course_end thiếu hoặc không parse được")
    if (ended < enrolled).any():
        raise ValueError("course_end sớm hơn enroll_time")
    return pd.DataFrame({phase: enrolled + (ended - enrolled) * ratio for phase, ratio in PHASES.items()})


def temporal_course_split(
    courses: pd.DataFrame,
    *,
    group_col: str = "course_root_id",
    start_col: str = "course_start",
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> pd.Series:
    """Chia course root theo lần mở đầu tiên; tie-break bằng ID, ổn định."""
    if not np.isclose(sum(fractions), 1.0) or any(value <= 0 for value in fractions):
        raise ValueError("fractions phải dương và có tổng 1")
    first = courses.groupby(group_col, as_index=False)[start_col].min()
    first[start_col] = pd.to_datetime(first[start_col], errors="coerce")
    if first[start_col].isna().any():
        raise ValueError("course_start thiếu/không parse được")
    first = first.sort_values([start_col, group_col], kind="mergesort").reset_index(drop=True)
    count = len(first)
    train_count = int(round(fractions[0] * count))
    dev_count = int(round(fractions[1] * count))
    split = np.full(count, "test", dtype=object)
    split[:train_count] = "train"
    split[train_count:train_count + dev_count] = "dev"
    return pd.Series(split, index=first[group_col].to_numpy(), name="split")


def assert_disjoint_groups(
    frame: pd.DataFrame,
    *,
    group_col: str = "course_root_id",
    split_col: str = "split",
) -> None:
    counts = frame.groupby(group_col)[split_col].nunique()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(f"{len(bad)} course root xuất hiện ở nhiều split")


def temporal_overlap_report(
    courses: pd.DataFrame,
    split: pd.Series,
    *,
    group_col: str = "course_root_id",
    start_col: str = "course_start",
    end_col: str = "course_end",
) -> dict[str, Any]:
    grouped = courses.groupby(group_col).agg({start_col: "min", end_col: "max"})
    grouped[start_col] = pd.to_datetime(grouped[start_col], errors="coerce")
    grouped[end_col] = pd.to_datetime(grouped[end_col], errors="coerce")
    if grouped[[start_col, end_col]].isna().any().any():
        raise ValueError("Timestamp course thiếu/không parse được")
    first_test_start = grouped.loc[split.index[split == "test"], start_col].min()
    train_end = grouped.loc[split.index[split == "train"], end_col]
    overlapping = train_end.index[train_end > first_test_start].tolist()
    return {
        "first_test_start": first_test_start.isoformat(),
        "train_course_count": int(len(train_end)),
        "overlap_count": int(len(overlapping)),
        "overlap_rate": float(len(overlapping) / len(train_end)),
        "overlapping_course_root_ids_private": overlapping,
    }


def choose_split_strategy(
    diagnostics: dict[str, Any],
    thresholds: SplitDecisionThresholds,
) -> dict[str, Any]:
    """TEMPORAL và SGKF là hai nhánh loại trừ, không chạy nối tiếp."""
    thresholds.validate()
    class_counts = diagnostics["temporal_test_course_counts_by_class"]
    reasons = {
        "timestamp_missing_rate_ok": float(diagnostics["timestamp_missing_rate"])
        <= thresholds.max_timestamp_missing_rate,
        "timestamp_order_ok": bool(diagnostics["timestamp_order_consistent"]),
        "test_courses_per_class_ok": min(int(class_counts.get(label, 0)) for label in CLASSES)
        >= thresholds.min_test_courses_per_class,
        "dev_ci_half_width_ok": float(diagnostics["dev_ci_half_width"])
        <= thresholds.max_dev_ci_half_width,
    }
    decision = "TEMPORAL_70_10_20" if all(reasons.values()) else "SGKF_5_FOLD_X_2"
    return {
        "schema_version": 1,
        "decision": decision,
        "reasons": reasons,
        "thresholds": asdict(thresholds),
        "diagnostics": diagnostics,
    }


def validate_pool_b(
    snapshots: pd.DataFrame,
    *,
    learner_col: str = "learner_id",
    course_col: str = "course_id",
) -> dict[str, Any]:
    """Pool B phải có đúng P1–P4 cho mỗi enrollment của khóa train."""
    required = {learner_col, course_col, "phase", "split", "q", "y"}
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise KeyError(f"Pool B thiếu cột: {missing}")
    if set(snapshots["split"].unique()) != {"train"}:
        raise ValueError("Pool B chỉ chứa enrollment thuộc khóa train")
    phases = snapshots.groupby([learner_col, course_col])["phase"].agg(
        lambda values: tuple(sorted(values)),
    )
    expected = tuple(sorted(PHASES))
    invalid = phases[phases != expected]
    if not invalid.empty:
        raise ValueError(f"{len(invalid)} enrollment không có đúng P1–P4")
    stable_labels = snapshots.groupby([learner_col, course_col])["y"].nunique()
    if (stable_labels != 1).any():
        raise ValueError("Nhãn enrollment phải cố định qua P1–P4")
    return {
        "schema_version": 1,
        "enrollments": int(len(phases)),
        "rows": int(len(snapshots)),
        "phases": list(PHASES),
        "strata": int(snapshots.groupby(["phase", "y"]).ngroups),
    }
