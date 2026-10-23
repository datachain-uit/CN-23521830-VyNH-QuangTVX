"""API tương thích Phụ lục B của Review 22/10/2026.

Các hàm thật nằm trong module giao thức/selector để pipeline và test không có hai
bản cài đặt khác nhau. Tệp này giữ đúng tên sản phẩm ``cq_patch_1022.py`` mà
giảng viên yêu cầu tại điểm kiểm tra 29/10.
"""

from __future__ import annotations

import numpy as np

from ..methods.controls import allocate_stratum_quotas
from ..methods.contracts import BudgetInfeasibleError
from .cq_v4_protocol import (
    build_course_families,
    concentration,
    split_course_families,
    temporal_overlap,
)


def course_families(ccid_sets: dict, min_ccid: int = 5, jaccard: float = 0.9) -> dict:
    return build_course_families(
        ccid_sets, threshold=jaccard, min_ccids=min_ccid,
    )


def family_split(family_rows, test_frac: float = 0.2, dev_frac: float = 0.1, seed: int = 42):
    records = family_rows.to_dict("records") if hasattr(family_rows, "to_dict") else family_rows
    normalized = []
    for record in records:
        item = dict(record)
        item["course_family_id"] = item.get("course_family_id", item.get("family_id"))
        item["family_start"] = item.get("family_start", item.get("start"))
        item["n_enroll_labeled"] = item.get("n_enroll_labeled", item.get("n_enroll"))
        normalized.append(item)
    return split_course_families(
        normalized, seed=seed, test_fraction=test_frac, dev_fraction=dev_frac,
    )


def sqrt_allocation(capacity: np.ndarray, m: int, floor: int = 20) -> np.ndarray:
    try:
        return allocate_stratum_quotas(
            np.asarray(capacity), int(m), policy="sqrt_floor", floor=int(floor),
        )
    except BudgetInfeasibleError as error:
        raise ValueError(str(error)) from error


def mass_weights(capacity: np.ndarray, quota: np.ndarray) -> np.ndarray:
    capacity = np.asarray(capacity, dtype=np.float64)
    quota = np.asarray(quota, dtype=np.float64)
    if capacity.shape != quota.shape or np.any(quota <= 0) or np.any(quota > capacity):
        raise ValueError("capacity/quota không hợp lệ")
    return capacity / quota


__all__ = [
    "concentration", "course_families", "family_split", "mass_weights",
    "sqrt_allocation", "temporal_overlap",
]
