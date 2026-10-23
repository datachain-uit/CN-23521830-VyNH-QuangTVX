"""Adapter BDIS gọi implementation AAAI 2022 của tác giả đã khóa commit.

Trạng thái: official-source adapter; thuật toán chọn representative và ngưỡng
``t1/t2`` đến từ source gốc, còn lớp bọc chỉ đổi dữ liệu sang contract chung và
vá alias ``numpy.int`` cho NumPy mới. Đây không phải tái lập nguyên vẹn mọi thiết
lập trong paper. Cơ chế: instance selection dựa trên cấu trúc mật độ/láng giềng
của BDIS. Loại đầu ra: tập con dòng thật với kích thước native do BDIS quyết định.

BDIS được đưa vào bài làm baseline chọn mẫu hiện đại và làm nguồn candidate cho
P03. Faiss là dependency bắt buộc; nếu thiếu, run phải BLOCKED thay vì âm thầm
đổi sang KMeans hay một thuật toán khác nhưng vẫn giữ tên BDIS.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseSelector
from ..contracts import exact_budget_size
from ..result import SelectionResult
from .common import verify_locked_repo


class BDISNativeSelector(BaseSelector):
    method_id = "n_bdis_native"

    def __init__(self, repo_root: Path, seed: int, options: dict[str, Any] | None = None):
        super().__init__(seed)
        self.repo = Path(repo_root) / "bdis"
        self.options = dict(options or {})
        self.structure_parent_ids_: np.ndarray | None = None
        self.candidate_mask_: np.ndarray | None = None

    def select(self, X_train, y_train, budget_ratio, **kwargs):
        requested = exact_budget_size(len(y_train), budget_ratio)
        source = self.repo / "algorithm.py"
        if not source.exists():
            return SelectionResult.failure(self.method_id, "blocked", requested, f"Thiếu repo: {self.repo}")
        if importlib.util.find_spec("faiss") is None:
            return SelectionResult.failure(
                self.method_id, "blocked", requested,
                "Thiếu faiss-cpu; BDIS native không được thay bằng sklearn",
            )
        try:
            upstream_commit = verify_locked_repo(self.repo)
        except RuntimeError as error:
            return SelectionResult.failure(self.method_id, "blocked", requested, str(error))
        started = time.perf_counter()
        had_np_int = hasattr(np, "int")
        old_np_int = getattr(np, "int", None)
        try:
            spec = importlib.util.spec_from_file_location("bdis_upstream_algorithm", source)
            module = importlib.util.module_from_spec(spec)
            # Upstream dùng np.int đã bị xóa ở NumPy mới; alias này là compatibility patch duy nhất.
            if not had_np_int:
                np.int = int  # type: ignore[attr-defined]
            assert spec.loader is not None
            spec.loader.exec_module(module)
            np.random.seed(self.seed)
            indices = np.asarray(module.BDIS(
                np.asarray(X_train, dtype=np.float32), np.asarray(y_train).reshape(-1, 1),
                int(self.options.get("t1", -1)), int(self.options.get("t2", 7)),
            ), dtype=np.int64).reshape(-1)
        except Exception as error:
            return SelectionResult.failure(self.method_id, "failed", requested, f"BDIS upstream lỗi: {error!r}")
        finally:
            if not had_np_int and hasattr(np, "int"):
                delattr(np, "int")
            elif had_np_int:
                np.int = old_np_int  # type: ignore[attr-defined]
        indices = np.unique(indices)
        self.candidate_mask_ = np.isin(np.arange(len(y_train)), indices)
        self.structure_parent_ids_ = np.asarray(y_train, dtype=np.int64)
        elapsed = time.perf_counter() - started
        return SelectionResult(
            indices=indices, weights=np.ones(len(indices)), requested_rows=requested,
            realized_rows=len(indices), budget_mode="native_realized", method_id=self.method_id,
            diagnostics={
                "upstream_repo": "https://github.com/CQQXY161120/Instance-Selection.git",
                "upstream_commit": upstream_commit,
                "reference_equivalent": True,
                "compatibility_patch": "temporary numpy.int alias only",
                "t1": int(self.options.get("t1", -1)), "t2": int(self.options.get("t2", 7)),
            },
            timings={"select": elapsed, "total": elapsed},
        )
