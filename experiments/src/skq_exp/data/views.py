"""Tạo selector view và learner view từ schema vai trò cột.

Selector hình học/kernel chỉ được thấy cột vai trò ``feature``. Learner có thể
nhận metadata và mã người học/giảng viên/trường nếu protocol cho phép từng cột;
course ID, target và target-derived vẫn bị loại. Hai view giữ nguyên row order để
selected indices tham chiếu trực tiếp về learner view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..artifacts import stable_hash


ColumnRole = Literal["id", "target", "target_derived", "metadata", "feature"]
VALID_ROLES = {"id", "target", "target_derived", "metadata", "feature"}
RESERVED_TARGET_DERIVED = {
    "q", "y", "qc", "yc", "coelo", "afelo", "acelo", "label", "label_f",
}


@dataclass(frozen=True)
class ColumnContract:
    name: str
    role: ColumnRole
    learner_allowed: bool = False
    selector_allowed: bool = False

    def validate(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"Vai trò cột không hợp lệ: {self.name}={self.role}")
        if self.selector_allowed and self.role != "feature":
            raise ValueError(f"Selector không được dùng cột {self.role}: {self.name}")
        if self.role in {"target", "target_derived"} and self.learner_allowed:
            raise ValueError(f"Learner không được dùng cột {self.role}: {self.name}")
        normalized = self.name.strip().lower()
        if normalized in RESERVED_TARGET_DERIVED and self.role not in {
            "target", "target_derived",
        }:
            raise ValueError(
                f"Cột dành riêng cho target/target-derived bị gán sai vai trò: {self.name}"
            )


@dataclass(frozen=True)
class FeatureViews:
    selector: pd.DataFrame
    learner: pd.DataFrame
    row_ids: np.ndarray
    contract_hash: str
    selector_columns: tuple[str, ...]
    learner_columns: tuple[str, ...]


def build_feature_views(
    frame: pd.DataFrame,
    contracts: tuple[ColumnContract, ...],
    *,
    row_ids: np.ndarray,
) -> FeatureViews:
    """Tạo hai view cùng row order và fail nếu schema thiếu/thừa cột."""
    for contract in contracts:
        contract.validate()
    names = [contract.name for contract in contracts]
    if len(names) != len(set(names)):
        raise ValueError("Feature contract có tên cột trùng")
    missing = sorted(set(names) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(names))
    if missing or unexpected:
        raise ValueError(f"Schema không khớp frame; missing={missing}, unexpected={unexpected}")
    row_ids = np.asarray(row_ids)
    if row_ids.ndim != 1 or len(row_ids) != len(frame):
        raise ValueError("row_ids phải là vector thẳng hàng với frame")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_ids phải duy nhất trong split")

    selector_columns = tuple(
        contract.name for contract in contracts if contract.selector_allowed
    )
    learner_columns = tuple(
        contract.name for contract in contracts if contract.learner_allowed
    )
    if not selector_columns or not learner_columns:
        raise ValueError("Selector view và learner view đều phải có ít nhất một cột")
    payload = {
        "schema_version": 1,
        "columns": [asdict(contract) for contract in contracts],
        "selector_columns": selector_columns,
        "learner_columns": learner_columns,
    }
    return FeatureViews(
        selector=frame.loc[:, selector_columns].copy(),
        learner=frame.loc[:, learner_columns].copy(),
        row_ids=row_ids.copy(),
        contract_hash=stable_hash(payload),
        selector_columns=selector_columns,
        learner_columns=learner_columns,
    )
