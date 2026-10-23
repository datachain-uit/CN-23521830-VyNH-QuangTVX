"""Hàm provenance dùng chung cho runner và freeze manifest.

Protocol hash bỏ đường dẫn/tài nguyên/chính sách lưu vì các trường đó không đổi
công thức thực nghiệm. Dataset fingerprint băm preprocessing manifest đã tạo từ
snapshot cụ thể, nhờ vậy freeze không thể dùng nhầm sang dữ liệu khác.
"""

from __future__ import annotations

from ..artifacts import sha256_file, stable_hash
from ..config import ExperimentConfig


def protocol_hash(config: ExperimentConfig) -> str:
    payload = config.to_dict()
    payload.pop("paths", None)
    payload.pop("resource", None)
    payload.pop("artifact_policy", None)
    return stable_hash(payload)


def dataset_fingerprint(config: ExperimentConfig, dataset_id: str) -> str:
    manifest = config.paths.processed_root / dataset_id / "preprocessing_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Thiếu preprocessing manifest: {manifest}")
    return sha256_file(manifest)
