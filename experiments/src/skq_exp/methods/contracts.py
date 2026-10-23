"""File này giữ các quy tắc budget và chỉ số dòng dùng chung.

Mục đích là để control, proposed và ablation đều bị kiểm tra giống nhau. Nếu không
thể phủ lớp trong budget đã yêu cầu, code báo lỗi rõ thay vì tự tăng budget.
"""

from __future__ import annotations

import numpy as np


class BudgetInfeasibleError(RuntimeError):
    pass


def exact_budget_size(n_samples: int, budget_ratio: float) -> int:
    if n_samples <= 0 or not 0 < budget_ratio <= 1:
        raise ValueError("n_samples và budget_ratio không hợp lệ")
    return max(1, int(round(n_samples * float(budget_ratio))))


def validate_exact_selection(indices: np.ndarray, n_samples: int, expected: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != expected:
        raise ValueError(f"Số dòng chọn được là {len(indices)}, cần đúng {expected}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Selection có dòng trùng")
    if indices.size and (indices.min() < 0 or indices.max() >= n_samples):
        raise ValueError("Selection chứa index ngoài train")
    return indices

