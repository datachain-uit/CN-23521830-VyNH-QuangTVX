"""File này định nghĩa đầu ra của phương pháp sinh dữ liệu bảng.

Khác coreset chọn dòng, kết quả chứa trực tiếp X/y mới và không giả lập
`selected_indices`. Kiểm tra shape, nhãn và trọng số được làm trước khi train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..result import VALID_STATUSES


@dataclass
class GeneratedDatasetResult:
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    requested_rows: int
    realized_rows: int
    generator_id: str
    output_type: str = "synthetic_raw"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "success"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Trạng thái không hợp lệ: {self.status}")
        if self.status != "success":
            return
        if self.X.ndim != 2 or self.y.ndim != 1 or len(self.X) != len(self.y):
            raise ValueError("Dữ liệu sinh phải có X 2-D và y 1-D cùng số dòng")
        if self.weights.shape != self.y.shape:
            raise ValueError("weights của dữ liệu sinh phải thẳng hàng với y")
        if self.realized_rows != len(self.y):
            raise ValueError("realized_rows không khớp dữ liệu sinh")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.weights)):
            raise ValueError("Dữ liệu sinh và trọng số phải hữu hạn")
        if np.any(self.weights < 0):
            raise ValueError("Trọng số dữ liệu sinh không được âm")

    @property
    def method_id(self) -> str:
        return self.generator_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.generator_id,
            "status": self.status,
            "requested_rows": self.requested_rows,
            "realized_rows": self.realized_rows,
            "output_type": self.output_type,
            "diagnostics": self.diagnostics,
            "timings": self.timings,
        }

    @classmethod
    def failure(cls, generator_id: str, status: str, requested_rows: int, reason: str):
        return cls(
            X=np.empty((0, 0), dtype=np.float32),
            y=np.empty(0, dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            requested_rows=requested_rows,
            realized_rows=0,
            generator_id=generator_id,
            diagnostics={"reason": reason},
            status=status,
        )
