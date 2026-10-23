"""Package này chuẩn bị dữ liệu trước khi selector được phép chạy.

Nó giữ nguồn, row ID, split và preprocessor trong manifest. Mọi phép fit imputer,
encoder hoặc scaler chỉ dùng train để tránh rò rỉ thông tin từ dev/test.
"""

from .adapters import get_dataset_adapter, prepare_external_dataset
from .prepare import fetch_dataset, load_processed, prepare_dataset
from .registry import DATASET_SPECS, get_dataset_spec

__all__ = [
    "DATASET_SPECS", "fetch_dataset", "get_dataset_spec", "prepare_dataset",
    "get_dataset_adapter", "prepare_external_dataset", "load_processed",
]
