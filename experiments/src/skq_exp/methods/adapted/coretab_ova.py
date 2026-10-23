"""CoreTab-XGB OVA adapted theo QĐ14 của Review 13/10/2026.

Đây không phải CoreTab native multiclass. Mỗi lớp gọi đúng adapter binary upstream
ở seed khác nhau, sau đó chỉ giữ positive rows tới quota lớp. Union không pad,
không lặp và giữ realized size nếu upstream trả thiếu positive.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseSelector
from ..contracts import exact_budget_size
from ..controls import allocate_largest_remainder
from ..native.coretab import CoreTabNativeSelector
from ..result import SelectionResult


class CoreTabXGBOVASelector(BaseSelector):
    method_id = "a_coretab_xgb_ova"

    def __init__(
        self,
        repo_root: Path,
        seed: int,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(seed)
        self.repo_root = Path(repo_root)
        self.options = dict(options or {})

    def select(
        self, X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float, **kwargs,
    ) -> SelectionResult:
        del kwargs
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self.validate_input(X_train, y_train, budget_ratio)
        requested = exact_budget_size(len(y_train), budget_ratio)
        labels, counts = np.unique(y_train, return_counts=True)
        if len(labels) < 3:
            return SelectionResult.failure(
                self.method_id, "na_contract", requested,
                "CoreTab-XGB OVA adapted chỉ dùng cho bài multiclass; binary dùng native",
            )
        quotas = allocate_largest_remainder(counts, requested, ensure_one=True)
        started = time.perf_counter()
        selected: list[np.ndarray] = []
        class_diagnostics = []
        upstream_commits = set()
        for class_position, (label, quota) in enumerate(zip(labels, quotas)):
            class_seed = self.seed + class_position
            binary_y = (y_train == label).astype(np.int64)
            binary = self._run_binary(
                X_train, binary_y, budget_ratio, class_seed,
            )
            if binary.status != "success":
                failure = SelectionResult.failure(
                    self.method_id,
                    binary.status,
                    requested,
                    f"CoreTab OVA lớp {label!r} lỗi: {binary.diagnostics.get('reason')}",
                )
                failure.diagnostics.update({
                    "adapted": True,
                    "failed_class": _plain(label),
                    "class_seed": class_seed,
                    "binary_status": binary.status,
                })
                return failure
            positive = np.asarray(binary.indices, dtype=np.int64)
            positive = positive[y_train[positive] == label]
            kept = positive[:int(quota)]
            selected.append(kept)
            commit = binary.diagnostics.get("upstream_commit")
            if commit:
                upstream_commits.add(str(commit))
            class_diagnostics.append({
                "label": _plain(label),
                "seed": class_seed,
                "quota": int(quota),
                "binary_returned_rows": int(binary.realized_rows),
                "positive_returned_rows": int(len(positive)),
                "kept_positive_rows": int(len(kept)),
                "binary_options": self._binary_options(class_position),
                "upstream_commit": commit,
            })
        indices = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
        unique = np.unique(indices)
        duplicates_removed = int(len(indices) - len(unique))
        if len(unique) == 0:
            return SelectionResult.failure(
                self.method_id, "budget_infeasible", requested,
                "CoreTab OVA không trả positive row cho bất kỳ lớp nào",
            )
        elapsed = time.perf_counter() - started
        return SelectionResult(
            indices=unique,
            weights=np.ones(len(unique), dtype=np.float64),
            requested_rows=requested,
            realized_rows=len(unique),
            budget_mode="adapted_ova_realized",
            method_id=self.method_id,
            diagnostics={
                "label": "CoreTab-XGB (OVA, adapted)",
                "adapted": True,
                "native_equivalent": False,
                "upstream_repo": "https://github.com/avivhadar33/coretab.git",
                "upstream_commits": sorted(upstream_commits),
                "quotas": quotas.tolist(),
                "class_runs": class_diagnostics,
                "duplicates_removed": duplicates_removed,
                "requested_ratio": float(budget_ratio),
                "realized_ratio": float(len(unique) / len(y_train)),
            },
            timings={"select": elapsed, "total": elapsed},
        )

    def _run_binary(
        self,
        X_train: np.ndarray,
        binary_y: np.ndarray,
        budget_ratio: float,
        class_seed: int,
    ) -> SelectionResult:
        selector = CoreTabNativeSelector(
            "n02_coretab_xgb_subset",
            self.repo_root,
            class_seed,
            self._binary_options(class_seed - self.seed),
        )
        return selector.select(X_train, binary_y, budget_ratio)

    def _binary_options(self, class_position: int) -> dict[str, Any]:
        base = dict(self.options.get("binary_options", {}))
        per_class = self.options.get("class_options", {})
        override = per_class.get(str(class_position), per_class.get(class_position, {}))
        return {**base, **dict(override)}


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
