"""Hợp đồng tạo family/split CQ V4 theo Review 22/10/2026.

Module này chỉ xử lý metadata nhóm và split; không đọc feature, không mở test và
không fit preprocessing. Nhóm dữ liệu có thể dùng cùng hàm để tạo manifest,
nhóm thực nghiệm dùng lại để kiểm toán kết quả bàn giao.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import numpy as np

from ..methods.controls import allocate_largest_remainder


def build_course_families(
    course_ccids: Mapping[Hashable, Iterable[Hashable]],
    *,
    threshold: float = 0.9,
    min_ccids: int = 5,
) -> dict[Hashable, str]:
    """Tạo connected components của đồ thị Jaccard-ccid."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold phải nằm trong (0, 1]")
    if min_ccids <= 0:
        raise ValueError("min_ccids phải dương")
    courses = sorted(course_ccids, key=lambda value: repr(value))
    sets = {course: frozenset(course_ccids[course]) for course in courses}
    parent = {course: course for course in courses}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if repr(root_left) <= repr(root_right):
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    eligible = [course for course in courses if len(sets[course]) >= min_ccids]
    for position, left in enumerate(eligible):
        left_set = sets[left]
        for right in eligible[position + 1:]:
            right_set = sets[right]
            union_size = len(left_set | right_set)
            similarity = len(left_set & right_set) / union_size if union_size else 0.0
            if similarity >= threshold:
                union(left, right)

    components: dict[Hashable, list[Hashable]] = defaultdict(list)
    for course in courses:
        components[find(course)].append(course)
    mapping: dict[Hashable, str] = {}
    for members in components.values():
        token = "\x1f".join(sorted((repr(item) for item in members)))
        family_id = f"cf_{hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]}"
        for course in members:
            mapping[course] = family_id
    return mapping


def split_course_families(
    family_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 42,
    test_fraction: float = 0.20,
    dev_fraction: float = 0.10,
) -> dict[str, str]:
    """Split family: test mới nhất; train/dev stratified ở phần còn lại."""

    rows = [dict(row) for row in family_rows]
    required = {"course_family_id", "family_start", "n_enroll_labeled", "has_E"}
    if not rows or any(required - set(row) for row in rows):
        raise ValueError(f"family_rows phải có đủ cột {sorted(required)}")
    ids = [str(row["course_family_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("course_family_id phải duy nhất trong family_rows")
    if not 0 < test_fraction < 1 or not 0 < dev_fraction < 1 - test_fraction:
        raise ValueError("test/dev fraction không hợp lệ")

    ordered = sorted(
        rows,
        key=lambda row: (str(row["family_start"]), str(row["course_family_id"])),
    )
    n_total = len(ordered)
    n_test = int(math.ceil(test_fraction * n_total))
    test_rows = ordered[-n_test:]
    remaining = ordered[:-n_test]
    desired_dev = min(len(remaining), int(round(dev_fraction * n_total)))

    sizes = np.asarray([float(row["n_enroll_labeled"]) for row in remaining])
    rank_order = np.argsort(sizes, kind="stable")
    quartile = np.empty(len(remaining), dtype=np.int64)
    if len(remaining):
        quartile[rank_order] = np.minimum(
            3, np.floor(4 * np.arange(len(remaining)) / len(remaining)).astype(np.int64),
        )
    strata: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(remaining):
        key = f"q{quartile[index]}_e{int(bool(row['has_E']))}"
        strata[key].append(str(row["course_family_id"]))
    stratum_keys = sorted(strata)
    capacities = np.asarray([len(strata[key]) for key in stratum_keys], dtype=np.int64)
    quotas = allocate_largest_remainder(capacities, desired_dev, ensure_one=False)
    rng = np.random.default_rng(seed)
    dev_ids: set[str] = set()
    for key, quota in zip(stratum_keys, quotas):
        candidates = np.asarray(sorted(strata[key]), dtype=object)
        if quota:
            dev_ids.update(rng.choice(candidates, size=int(quota), replace=False).tolist())

    result = {
        str(row["course_family_id"]): (
            "dev" if str(row["course_family_id"]) in dev_ids else "train"
        )
        for row in remaining
    }
    result.update({str(row["course_family_id"]): "test" for row in test_rows})
    if len(result) != n_total:
        raise RuntimeError("Family split làm mất hoặc trùng family")
    return result


def temporal_overlap(
    family_rows: Iterable[Mapping[str, Any]],
    split_by_family: Mapping[str, str],
) -> dict[str, Any]:
    """Tỷ lệ family train còn hoạt động sau thời điểm test bắt đầu sớm nhất."""

    rows = [dict(row) for row in family_rows]
    test_starts = [
        str(row["family_start"]) for row in rows
        if split_by_family[str(row["course_family_id"])] == "test"
    ]
    if not test_starts:
        raise ValueError("Split không có test family")
    boundary = min(test_starts)
    train = [
        row for row in rows
        if split_by_family[str(row["course_family_id"])] == "train"
    ]
    overlap_ids = sorted(
        str(row["course_family_id"]) for row in train
        if str(row.get("family_end", row["family_start"])) >= boundary
    )
    denominator = len(train)
    ratio = len(overlap_ids) / denominator if denominator else 0.0
    return {
        "test_start_boundary": boundary,
        "overlap_family_count": len(overlap_ids),
        "train_family_count": denominator,
        "overlap_ratio": ratio,
        "overlap_family_ids": overlap_ids,
        "requires_nolfo": ratio > 0.05,
    }


def concentration(group_ids: Iterable[Hashable]) -> dict[str, float | int]:
    """Largest share và effective number ``1/sum(p_g^2)`` của một split."""

    values, counts = np.unique(np.asarray(list(group_ids), dtype=object), return_counts=True)
    if not len(values):
        raise ValueError("Không thể tính concentration trên split rỗng")
    shares = counts.astype(np.float64) / counts.sum()
    return {
        "n_rows": int(counts.sum()),
        "n_families": int(len(values)),
        "largest_share": float(shares.max()),
        "effective_families": float(1.0 / np.sum(shares ** 2)),
    }
