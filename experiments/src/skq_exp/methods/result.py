"""File này định nghĩa đầu ra chung của mọi phương pháp chọn dòng thật.

Đối tượng giữ chỉ số dòng, trọng số, budget, thời gian và chẩn đoán trong cùng một
nơi. Nó kiểm tra lỗi ngay khi được tạo để một artifact sai không đi tiếp sang bước
train model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


VALID_STATUSES = {
    "success", "failed", "oom", "predicted_oom", "predicted_timeout", "timeout", "blocked",
    "budget_infeasible", "na_contract", "structure_collapsed", "storage_limit",
    "gate_locked",
}


@dataclass
class SelectionResult:
    indices: np.ndarray
    weights: np.ndarray
    requested_rows: int
    realized_rows: int
    budget_mode: str
    method_id: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "success"

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Trạng thái không hợp lệ: {self.status}")
        if self.status != "success":
            return
        if self.indices.ndim != 1 or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("indices phải là mảng 1-D và không được trùng")
        if self.weights.shape != self.indices.shape:
            raise ValueError("weights phải cùng chiều với indices")
        if np.any(self.weights < 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError("weights phải hữu hạn và không âm")
        if self.realized_rows != len(self.indices):
            raise ValueError("realized_rows không khớp số indices")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "status": self.status,
            "requested_rows": self.requested_rows,
            "realized_rows": self.realized_rows,
            "budget_mode": self.budget_mode,
            "diagnostics": self.diagnostics,
            "timings": self.timings,
        }

    @classmethod
    def failure(
        cls, method_id: str, status: str, requested_rows: int, reason: str,
    ) -> "SelectionResult":
        return cls(
            indices=np.empty(0, dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            requested_rows=requested_rows,
            realized_rows=0,
            budget_mode="exact_total",
            method_id=method_id,
            diagnostics={"reason": reason},
            status=status,
        )

