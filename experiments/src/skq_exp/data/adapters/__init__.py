"""Các adapter biến dữ liệu ngoài thành processed contract thống nhất."""

from .base import DatasetAdapter
from .presplit_csv import PresplitCSVAdapter, PresplitCSVLayout, TrainOnlyOrdinalCSVAdapter
from .registry import DATASET_ADAPTERS, get_dataset_adapter, prepare_external_dataset

__all__ = [
    "DATASET_ADAPTERS",
    "DatasetAdapter",
    "PresplitCSVAdapter",
    "PresplitCSVLayout",
    "TrainOnlyOrdinalCSVAdapter",
    "get_dataset_adapter",
    "prepare_external_dataset",
]
