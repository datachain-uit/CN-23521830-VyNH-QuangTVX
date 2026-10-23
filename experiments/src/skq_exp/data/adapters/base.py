"""Contract chung cho mọi adapter dữ liệu ngoài của SKQ.

Adapter chỉ chịu trách nhiệm đổi snapshot nguồn thành contract processed cố định.
Runner phía sau luôn đọc cùng các file ``X/y/row_ids`` nên không cần biết CSV,
Parquet, nhiều temporal snapshot hay schema gốc của từng dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetAdapter(ABC):
    """Giao diện tối thiểu để thêm một dataset không do ``skq fetch-data`` tải."""

    dataset_id: str

    @abstractmethod
    def prepare(
        self,
        input_dir: str | Path,
        processed_root: str | Path,
        *,
        overwrite: bool = False,
        validate_expected_rows: bool = True,
    ) -> Path:
        """Tạo processed artifact và trả đường dẫn preprocessing manifest."""
