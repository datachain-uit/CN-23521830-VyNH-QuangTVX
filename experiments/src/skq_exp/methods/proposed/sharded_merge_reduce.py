"""Hai biến thể SKQ phân mảnh để giảm đỉnh RAM trên dữ liệu rất lớn.

Trạng thái: P10/P11 là **đề xuất của dự án**, không phải phương pháp native và
không tuyên bố tương đương nguyên paper CoreTab/BDIS. Mỗi dòng train được đưa
vào đúng một shard xác định; source chính thức chỉ tạo structure/candidate trong
từng shard, sau đó SKQ cấp budget toàn cục và chạy RFF-herding/QP theo từng nhóm.

P10 dùng leaf structure CoreTab-XGB và cho mọi dòng trong shard làm candidate,
vì mục tiêu là giữ exact 5% giống P02. P11 giữ đúng candidate do BDIS trả về;
nếu pool nhỏ hơn 5% thì báo realized size thay vì pad hoặc lặp dòng. Hai biến thể
không dùng proxy dataset và đầu ra vẫn là subset dòng thật có trọng số.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..base import BaseSelector
from ..contracts import exact_budget_size
from ..native import BDISNativeSelector, CoreTabNativeSelector
from ..result import SelectionResult
from .structured_kquad import StructuredKQuadSelector


ParentFactory = Callable[[str, int, dict[str, Any]], BaseSelector]


class ShardedMergeReduceSelector(BaseSelector):
    """Tạo parent structure theo shard rồi reduce toàn cục bằng SKQ streaming."""

    def __init__(
        self,
        method_id: str,
        repo_root: Path,
        seed: int = 11,
        options: dict[str, Any] | None = None,
        n_components: int = 256,
        parent_factory: ParentFactory | None = None,
    ):
        super().__init__(seed)
        self.method_id = method_id
        self.repo_root = Path(repo_root)
        self.options = dict(options or {})
        self.n_components = int(n_components)
        self.parent_factory = parent_factory

    def _build_parent(self, parent_method: str, shard_seed: int, options: dict[str, Any]):
        if self.parent_factory is not None:
            return self.parent_factory(parent_method, shard_seed, options)
        if parent_method == "n02_coretab_xgb_subset":
            return CoreTabNativeSelector(parent_method, self.repo_root, shard_seed, options)
        if parent_method == "n_bdis_native":
            return BDISNativeSelector(self.repo_root, shard_seed, options)
        raise ValueError(f"Parent phân mảnh chưa hỗ trợ: {parent_method}")

    def select(self, X_train, y_train, budget_ratio, **kwargs):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self.validate_input(X_train, y_train, budget_ratio)
        requested = exact_budget_size(len(y_train), budget_ratio)
        started = time.perf_counter()
        shard_rows = int(self.options.get("shard_rows", 10_000))
        if shard_rows < 2:
            return SelectionResult.failure(
                self.method_id, "failed", requested, "shard_rows phải từ 2 trở lên"
            )
        parent_method = str(self.options.get("parent_method", ""))
        expected_parent = (
            "n02_coretab_xgb_subset"
            if self.method_id == "p10_skq_mr_coretab_xgb"
            else "n_bdis_native"
        )
        if parent_method != expected_parent:
            return SelectionResult.failure(
                self.method_id,
                "blocked",
                requested,
                f"{self.method_id} yêu cầu parent_method={expected_parent}",
            )
        parent_options = dict(self.options.get("parent_options", {}))
        shards = _stratified_shards(y_train, shard_rows, self.seed)
        parent_ids = np.empty(len(y_train), dtype=np.int64)
        candidate_mask = np.zeros(len(y_train), dtype=bool)
        next_parent = 0
        collapsed_parent_groups = 0
        parent_statuses: list[str] = []
        upstream_commits: set[str] = set()
        parent_seconds = 0.0

        for shard_index, rows in enumerate(shards):
            if kwargs.get("resource_guard") is not None:
                kwargs["resource_guard"].check("sharded_parent")
            parent = self._build_parent(
                parent_method, self.seed + shard_index, parent_options
            )
            parent_started = time.perf_counter()
            result = parent.select(X_train[rows], y_train[rows], budget_ratio)
            parent_seconds += time.perf_counter() - parent_started
            parent_statuses.append(result.status)
            commit = result.diagnostics.get("upstream_commit")
            if commit:
                upstream_commits.add(str(commit))
            local_parent = getattr(parent, "structure_parent_ids_", None)
            if local_parent is None:
                return SelectionResult.failure(
                    self.method_id,
                    result.status if result.status != "success" else "failed",
                    requested,
                    f"Parent {parent_method} shard {shard_index} không xuất structure: "
                    f"{result.diagnostics.get('reason', result.status)}",
                )
            _, local_encoded = np.unique(
                np.asarray(local_parent), return_inverse=True, axis=0
            )
            local_encoded = np.asarray(local_encoded, dtype=np.int64)
            if parent_method == "n02_coretab_xgb_subset":
                before = int(local_encoded.max(initial=-1)) + 1
                local_encoded = _coarsen_parent_groups(
                    local_encoded,
                    int(self.options.get("max_parent_groups_per_shard", 64)),
                )
                collapsed_parent_groups += before - (
                    int(local_encoded.max(initial=-1)) + 1
                )
            parent_ids[rows] = local_encoded + next_parent
            next_parent += int(local_encoded.max(initial=-1)) + 1

            if parent_method == "n_bdis_native":
                local_candidates = getattr(parent, "candidate_mask_", None)
                if local_candidates is None:
                    return SelectionResult.failure(
                        self.method_id, "failed", requested,
                        f"BDIS shard {shard_index} không xuất candidate_mask",
                    )
                candidate_mask[rows] = np.asarray(local_candidates, dtype=bool)
            else:
                # P10 dùng CoreTab để tạo leaf structure, giống P02; không giới
                # hạn candidate theo native realized size để giữ exact budget.
                candidate_mask[rows] = True

        engine = StructuredKQuadSelector(
            self.method_id,
            seed=self.seed,
            n_components=self.n_components,
            bandwidth_multiplier=float(self.options.get("bandwidth_multiplier", 1.0)),
            budget_policy=str(self.options.get("budget_policy", "exact_total")),
            stream_groups=True,
        )
        result = engine.select(
            X_train,
            y_train,
            budget_ratio,
            parent_ids=parent_ids,
            candidate_mask=candidate_mask,
            row_ids=kwargs.get("row_ids"),
            resource_guard=kwargs.get("resource_guard"),
            resource_policy=kwargs.get("resource_policy"),
        )
        result.timings["parent_shards"] = parent_seconds
        result.timings["skq_reduce"] = float(result.timings.get("total", 0.0))
        result.timings["total"] = time.perf_counter() - started
        result.diagnostics.update({
            "variant_family": "sharded_merge_reduce",
            "parent_method": parent_method,
            "parent_statuses": parent_statuses,
            "upstream_commits": sorted(upstream_commits),
            "shard_rows_limit": shard_rows,
            "shard_count": len(shards),
            "largest_shard_rows": max(map(len, shards)),
            "all_train_rows_seen": True,
            "train_rows_seen": int(sum(map(len, shards))),
            "proxy_dataset": False,
            "output_rows_are_original_train_rows": True,
            "candidate_rows_after_shards": int(candidate_mask.sum()),
            "parent_groups_after_shards": int(next_parent),
            "collapsed_parent_groups": int(collapsed_parent_groups),
            "native_equivalence": False,
            "declared_adaptation": (
                "official parent run independently per deterministic shard; "
                "global SKQ allocation/RFF-herding/simplex-QP"
            ),
        })
        return result


def _stratified_shards(y: np.ndarray, max_rows: int, seed: int) -> list[np.ndarray]:
    """Chia shard cân bằng, xác định và dùng mỗi dòng đúng một lần."""
    y = np.asarray(y)
    shard_count = max(1, math.ceil(len(y) / max_rows))
    buckets: list[list[int]] = [[] for _ in range(shard_count)]
    rng = np.random.default_rng(seed)
    cursor = 0
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        rows = rows[rng.permutation(len(rows))]
        for row in rows:
            buckets[cursor % shard_count].append(int(row))
            cursor += 1
    shards = [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]
    if any(len(shard) > max_rows for shard in shards):
        raise RuntimeError("Bộ chia shard vượt shard_rows")
    if not np.array_equal(
        np.sort(np.concatenate(shards)), np.arange(len(y), dtype=np.int64)
    ):
        raise RuntimeError("Bộ chia shard không phủ đúng toàn bộ train")
    return shards


def _coarsen_parent_groups(parent_ids: np.ndarray, max_groups: int) -> np.ndarray:
    """Giữ nhóm leaf lớn và gộp leaf nhỏ vào residual trong cùng shard.

    Đây là bước merge minh bạch của P10, không phải thay CoreTab bằng proxy.
    Nó chặn số QP con tăng tới số dòng khi tổ hợp leaf XGBoost quá phân mảnh.
    """
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    if max_groups < 2:
        raise ValueError("max_parent_groups_per_shard phải từ 2 trở lên")
    groups, counts = np.unique(parent_ids, return_counts=True)
    if len(groups) <= max_groups:
        _, encoded = np.unique(parent_ids, return_inverse=True)
        return encoded.astype(np.int64)
    order = np.lexsort((groups, -counts))
    kept = groups[order[:max_groups - 1]]
    output = np.full(len(parent_ids), max_groups - 1, dtype=np.int64)
    for new_id, group in enumerate(kept):
        output[parent_ids == group] = new_id
    return output
