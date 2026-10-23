"""Các ablation nội bộ dùng để giải thích thành phần nào của SKQ có tác dụng.

Trạng thái: code của đề tài, không phải baseline paper độc lập. D02 thay herding
bằng random nhưng giữ phân nhóm; D04 bỏ parent structure và chỉ giữ lớp; D05 giữ
nguyên các dòng của winner nhưng thay QP bằng trọng số đều trong nhóm. Tất cả đều
trả subset dòng thật. Chúng được đưa vào bài để mỗi phép so sánh chỉ thay một thành
phần, từ đó tách đóng góp của structure, kernel herding và tối ưu trọng số.
"""

from __future__ import annotations

import time

import numpy as np

from .proposed.structured_kquad import StructuredKQuadSelector
from .result import SelectionResult


def build_d02(seed: int, n_components: int = 256) -> StructuredKQuadSelector:
    return StructuredKQuadSelector(
        "d02_parent_structured_random", seed=seed, n_components=n_components,
        random_within_group=True, use_qp=True,
    )


def build_d04(seed: int, n_components: int = 256) -> StructuredKQuadSelector:
    return StructuredKQuadSelector(
        "d04_global_rff_quadrature", seed=seed, n_components=n_components,
        random_within_group=False, use_qp=True,
    )


def equal_group_weights(
    base: SelectionResult,
    full_group_ids: np.ndarray,
) -> SelectionResult:
    if base.status != "success":
        return SelectionResult.failure(
            "d05_equal_group_weight", "blocked", base.requested_rows,
            "D05 cần selection thành công của frozen winner",
        )
    started = time.perf_counter()
    full_group_ids = np.asarray(full_group_ids)
    selected_groups = full_group_ids[base.indices]
    weights = np.empty(len(base.indices), dtype=np.float64)
    for group in np.unique(selected_groups):
        selected_mask = selected_groups == group
        population = int(np.sum(full_group_ids == group))
        weights[selected_mask] = population / int(selected_mask.sum())
    reweight_time = time.perf_counter() - started
    timings = dict(base.timings)
    source_total = float(timings.get("total", 0.0))
    timings.update({"reweight": reweight_time, "source_selection": source_total, "total": source_total + reweight_time})
    return SelectionResult(
        indices=base.indices.copy(),
        weights=weights,
        requested_rows=base.requested_rows,
        realized_rows=base.realized_rows,
        budget_mode=base.budget_mode,
        method_id="d05_equal_group_weight",
        diagnostics={"source_method": base.method_id, "same_indices": True},
        timings=timings,
    )
