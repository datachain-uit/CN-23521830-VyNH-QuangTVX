"""File này kiểm tra sample weight trước khi train model.

Nó xác nhận trọng số thẳng hàng với dòng được chọn, hữu hạn, không âm và còn tổng
dương. Class weight được tính từ full train rồi nhân với method weight; không tính
lại từ coreset vì việc đó sẽ thay đổi mục tiêu giữa các phương pháp. Trọng số hiệu
dụng cuối cùng luôn được chuẩn hóa về mean=1 theo QĐ15/O11/O14 của Review 13.
"""

from __future__ import annotations

import numpy as np


def balanced_class_weights(y_full_train: np.ndarray) -> dict[int, float]:
    labels, counts = np.unique(y_full_train, return_counts=True)
    total = len(y_full_train)
    return {int(label): total / (len(labels) * count) for label, count in zip(labels, counts)}


def effective_sample_weights(
    selected_y: np.ndarray,
    method_weights: np.ndarray,
    class_weights: dict[int, float] | None,
) -> np.ndarray:
    method_weights = np.asarray(method_weights, dtype=np.float64)
    if method_weights.shape != np.asarray(selected_y).shape:
        raise ValueError("method_weights không thẳng hàng với selected_y")
    multipliers = np.ones(len(selected_y), dtype=np.float64)
    if class_weights:
        multipliers = np.asarray([class_weights[int(label)] for label in selected_y])
    result = method_weights * multipliers
    if np.any(result < 0) or not np.all(np.isfinite(result)) or result.sum() <= 0:
        raise ValueError("effective sample weight không hợp lệ")
    # Chuẩn hóa chỉ đổi thang chung của loss, không đổi tỷ lệ tương đối giữa mẫu.
    # Mean=1 giúp FullTrain, subset thật và synthetic dùng cùng một quy ước loss.
    return result * (len(result) / result.sum())


def sample_weight_diagnostics(weights: np.ndarray) -> dict[str, float | int]:
    """Trả thống kê có thể audit mà không ghi toàn bộ vector trọng số."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("weights phải là vector một chiều không rỗng")
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "sum": float(values.sum()),
        "zero_count": int(np.count_nonzero(values == 0)),
    }

