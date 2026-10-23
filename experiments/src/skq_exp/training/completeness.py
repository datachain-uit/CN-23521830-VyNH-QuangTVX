"""Kiểm mẫu số run trước mọi suy luận thống kê.

Run lỗi/timeout/blocked không bị xóa và cũng không bị gán metric bằng 0. Nếu một
ô method×seed×phase thiếu hoặc không success, bảng mô tả vẫn được phép xuất với
tỷ lệ hoàn tất, nhưng kiểm định được đánh dấu ``not_estimable_incomplete``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def audit_run_completeness(
    rows: Iterable[Mapping[str, Any]],
    expected_cells: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str] = ("method_id", "selector_seed", "phase"),
) -> dict[str, Any]:
    """Audit đủ ô và status; không inner-join hoặc drop run thất bại."""
    fields = tuple(str(field) for field in key_fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("key_fields phải khác rỗng và không trùng")
    expected = [_key(cell, fields) for cell in expected_cells]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_cells phải khác rỗng và không trùng")

    observed: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    unexpected = []
    expected_set = set(expected)
    for row in rows:
        key = _key(row, fields)
        if key not in expected_set:
            unexpected.append(_render_key(key, fields))
            continue
        observed.setdefault(key, []).append(row)

    missing = [key for key in expected if key not in observed]
    duplicates = [key for key, values in observed.items() if len(values) != 1]
    status_counts: Counter[str] = Counter()
    non_success = []
    success_count = 0
    for key in expected:
        values = observed.get(key, [])
        if len(values) != 1:
            continue
        status = str(values[0].get("status", "missing_status"))
        status_counts[status] += 1
        if status == "success":
            success_count += 1
        else:
            non_success.append({
                **_render_key(key, fields),
                "status": status,
                "reason": values[0].get("reason")
                or dict(values[0].get("diagnostics") or {}).get("reason"),
            })

    denominator = len(expected)
    ready = not missing and not duplicates and not unexpected and not non_success
    return {
        "schema_version": 1,
        "key_fields": list(fields),
        "expected_denominator": denominator,
        "successful_cells": success_count,
        "completion_rate": success_count / denominator,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_cells": [_render_key(key, fields) for key in missing],
        "duplicate_cells": [_render_key(key, fields) for key in duplicates],
        "unexpected_cells": unexpected,
        "non_success_cells": non_success,
        "failed_runs_kept_in_denominator": True,
        "failed_metric_imputation": "forbidden",
        "inferential_ready": ready,
        "inference_status": "ready" if ready else "not_estimable_incomplete",
    }


def _key(row: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    missing = [field for field in fields if field not in row]
    if missing:
        raise KeyError(f"Run/cell thiếu key fields: {missing}")
    return tuple(row[field] for field in fields)


def _render_key(key: tuple[Any, ...], fields: tuple[str, ...]) -> dict[str, Any]:
    return dict(zip(fields, key))
