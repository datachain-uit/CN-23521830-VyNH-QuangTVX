"""Boundary nhập kết quả AutoCoreset từ source chính thức đã khóa commit.

Trạng thái: official-source adapter chạy qua driver riêng; thuật toán upstream
được giữ nguyên nhưng có vá tương thích API NumPy/sklearn, nên không tuyên bố môi
trường chạy giống tuyệt đối paper. Cơ chế: AutoCoreset tự động xây coreset có
trọng số cho bài toán logistic. Loại đầu ra: index dòng thật và trọng số native.

AutoCoreset được đưa vào bài làm baseline coreset có trọng số và tối ưu tự động.
File này không tự thay thuật toán: nó chỉ nhận artifact khi commit, fingerprint,
index và trọng số đều qua gate. Thiếu artifact hợp lệ phải trả BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseSelector
from ..contracts import exact_budget_size
from ..result import SelectionResult
from .common import verify_locked_repo


class AutoCoresetNativeSelector(BaseSelector):
    method_id = "n_autocoreset_native"

    def __init__(self, repo_root: Path, seed: int, options: dict[str, Any] | None = None):
        super().__init__(seed)
        self.repo = Path(repo_root) / "autocoreset"
        self.options = dict(options or {})

    def select(self, X_train, y_train, budget_ratio, **kwargs):
        requested = exact_budget_size(len(y_train), budget_ratio)
        artifact_dir = Path(self.options.get("native_artifact_dir", ""))
        if not artifact_dir.is_dir():
            return SelectionResult.failure(
                self.method_id, "blocked", requested,
                "AutoCoreset cần artifact từ environment reproduction riêng; "
                "hãy chạy scripts/run_autocoreset_native.py theo GUIDELINE",
            )
        try:
            manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
            indices = np.load(artifact_dir / "indices.npy")
            weights = np.load(artifact_dir / "weights.npy")
        except Exception as error:
            return SelectionResult.failure(self.method_id, "failed", requested, f"Artifact AutoCoreset lỗi: {error!r}")
        try:
            expected_commit = verify_locked_repo(self.repo)
        except RuntimeError as error:
            return SelectionResult.failure(self.method_id, "blocked", requested, str(error))
        if manifest.get("upstream_commit") != expected_commit:
            return SelectionResult.failure(self.method_id, "blocked", requested, "Commit artifact AutoCoreset không khớp lock")
        if manifest.get("dataset_fingerprint") != self.options.get("dataset_fingerprint"):
            return SelectionResult.failure(self.method_id, "blocked", requested, "Fingerprint dữ liệu AutoCoreset không khớp")
        indices = np.asarray(indices, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        if (
            indices.ndim != 1 or len(indices) != len(weights)
            or len(np.unique(indices)) != len(indices)
            or (indices.size and (indices.min() < 0 or indices.max() >= len(y_train)))
            or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0
        ):
            return SelectionResult.failure(
                self.method_id, "failed", requested,
                "Artifact AutoCoreset vi phạm index/weight contract",
            )
        return SelectionResult(
            indices=indices, weights=weights, requested_rows=requested,
            realized_rows=len(indices), budget_mode="native_realized", method_id=self.method_id,
            diagnostics={**manifest, "reference_equivalent": True},
            timings=manifest.get("timings", {}),
        )
