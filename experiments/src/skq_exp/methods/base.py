"""File này mô tả giao diện tối thiểu mà một selector phải tuân theo.

Selector nhận duy nhất dữ liệu train cùng budget và trả SelectionResult. Việc giữ
giao diện nhỏ giúp test leakage dễ hơn và tránh một method vô tình nhận dev/test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .result import SelectionResult


class BaseSelector(ABC):
    method_id: str

    def __init__(self, seed: int = 11):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def select(
        self, X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float, **kwargs,
    ) -> SelectionResult:
        raise NotImplementedError

    @staticmethod
    def validate_input(X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float) -> None:
        if X_train.ndim != 2 or y_train.ndim != 1 or len(X_train) != len(y_train):
            raise ValueError("X_train phải là 2-D, y_train là 1-D và có cùng số dòng")
        if not 0 < budget_ratio <= 1:
            raise ValueError("budget_ratio phải nằm trong (0, 1]")

