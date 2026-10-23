"""Registry tường minh cho các dataset adapter ngoài.

Muốn thêm dataset: tạo một class kế thừa ``DatasetAdapter`` trong thư mục này,
thêm instance vào ``DATASET_ADAPTERS`` và thêm metadata vào data/registry.py.
Registry tường minh giúp audit chính xác adapter nào đã tạo artifact.
"""

from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter
from .course_quality import CourseQualityMEDAdapter
from .course_quality_v1 import CourseQualityV1Adapter, CourseQualityV1Afelo01Adapter
from .course_quality_v4 import CourseQualityV4Adapter


DATASET_ADAPTERS: dict[str, DatasetAdapter] = {
    adapter.dataset_id: adapter
    for adapter in (
        CourseQualityMEDAdapter(),
        CourseQualityV1Adapter(),
        CourseQualityV1Afelo01Adapter(),
        CourseQualityV4Adapter(),
    )
}


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    try:
        return DATASET_ADAPTERS[dataset_id]
    except KeyError as error:
        raise KeyError(
            f"Dataset {dataset_id} chưa có external adapter; "
            "dùng skq prepare cho dataset built-in hoặc đăng ký adapter mới"
        ) from error


def prepare_external_dataset(
    dataset_id: str,
    input_dir: str | Path,
    processed_root: str | Path,
    *,
    overwrite: bool = False,
    validate_expected_rows: bool = True,
) -> Path:
    return get_dataset_adapter(dataset_id).prepare(
        input_dir,
        processed_root,
        overwrite=overwrite,
        validate_expected_rows=validate_expected_rows,
    )
