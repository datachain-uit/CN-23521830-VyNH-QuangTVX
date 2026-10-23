"""CRAIG feature-space dùng solver facility-location từ repo tác giả.

Trạng thái: adaptation mức R1, không phải CRAIG native và không được ghi là tái
hiện y hệt paper. Code lấy ``FacilityLocation`` và lazy-greedy từ source chính
thức, nhưng xây ma trận khoảng cách trực tiếp trong không gian feature thay vì
pipeline gradient logistic đầy đủ của CRAIG. Cơ chế: chọn đại diện theo lớp bằng
facility-location; trọng số là số điểm được gán cho mỗi đại diện. Loại đầu ra:
tập con dòng thật có trọng số.

Phương pháp được đưa vào bài để so sánh SKQ với họ submodular/facility-location,
đồng thời minh bạch giới hạn tái hiện. Do pairwise O(n²), runner phải qua gate
RAM/số phép tính; OOM là kết quả tài nguyên chứ không được đổi thuật toán.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances

from ..base import BaseSelector
from ..contracts import exact_budget_size
from ..result import SelectionResult
from .common import verify_locked_repo


class CRAIGNativeSelector(BaseSelector):
    method_id = "a_craig_feature_space"

    def __init__(self, repo_root: Path, seed: int, options: dict[str, Any] | None = None):
        super().__init__(seed)
        self.repo = Path(repo_root) / "craig"
        self.options = dict(options or {})

    def select(self, X_train, y_train, budget_ratio, **kwargs):
        requested = exact_budget_size(len(y_train), budget_ratio)
        source = self.repo / "lazy_greedy.py"
        if not source.exists():
            return SelectionResult.failure(self.method_id, "blocked", requested, f"Thiếu repo: {self.repo}")
        try:
            upstream_commit = verify_locked_repo(self.repo)
        except RuntimeError as error:
            return SelectionResult.failure(self.method_id, "blocked", requested, str(error))
        guard = kwargs.get("resource_guard")
        started = time.perf_counter()
        try:
            spec = importlib.util.spec_from_file_location("craig_upstream_lazy_greedy", source)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            selected, weights = [], []
            labels, counts = np.unique(y_train, return_counts=True)
            quotas = np.ceil(counts / len(y_train) * requested).astype(int)
            for label, quota in zip(labels, quotas):
                pool = np.flatnonzero(np.asarray(y_train) == label)
                if guard is not None:
                    guard.check(f"craig_pairwise_class_{label}")
                distances = pairwise_distances(np.asarray(X_train)[pool], metric="euclidean", n_jobs=1)
                similarities = distances.max() - distances
                objective = module.FacilityLocation(similarities, list(range(len(pool))))
                local, _ = module.lazy_greedy_heap(objective, list(range(len(pool))), min(int(quota), len(pool)))
                local = np.asarray(local, dtype=np.int64)
                chosen = pool[local]
                selected.extend(chosen.tolist())
                assignment = np.argmax(similarities[:, local], axis=1)
                weights.extend(np.bincount(assignment, minlength=len(local)).astype(float).tolist())
                del distances, similarities
        except MemoryError:
            return SelectionResult.failure(self.method_id, "oom", requested, "CRAIG hết bộ nhớ ở ma trận pairwise")
        except Exception as error:
            return SelectionResult.failure(self.method_id, "failed", requested, f"CRAIG upstream lỗi: {error!r}")
        elapsed = time.perf_counter() - started
        indices = np.asarray(selected, dtype=np.int64)
        return SelectionResult(
            indices=indices, weights=np.asarray(weights), requested_rows=requested,
            realized_rows=len(indices), budget_mode="native_realized", method_id=self.method_id,
            diagnostics={
                "upstream_repo": "https://github.com/baharanm/craig.git",
                "upstream_commit": upstream_commit,
                "reference_equivalent": False,
                "fidelity_tier": "R1_selection_stage_adaptation",
                "pipeline": "feature-space classwise facility location using upstream lazy greedy",
                "class_quotas_ceil": quotas.tolist(),
            },
            timings={"select": elapsed, "total": elapsed},
        )
