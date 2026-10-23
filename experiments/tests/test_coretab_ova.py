"""CoreTab OVA adapted phải giữ quota/positive/realized và nhãn adapted."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skq_exp.methods.adapted.coretab_ova import CoreTabXGBOVASelector
from skq_exp.methods.result import SelectionResult


def test_coretab_ova_union_keeps_only_positive_without_padding(monkeypatch) -> None:
    X = np.arange(90, dtype=np.float32).reshape(30, 3)
    y = np.repeat([0, 1, 2], 10)
    selector = CoreTabXGBOVASelector(Path("unused"), seed=11)

    def _binary(_X, binary_y, budget_ratio, class_seed):
        del _X, budget_ratio
        positive = np.flatnonzero(binary_y == 1)
        negative = np.flatnonzero(binary_y == 0)
        keep_positive = positive[:2] if class_seed == 12 else positive[:4]
        indices = np.concatenate([keep_positive, negative[:3]])
        return SelectionResult(
            indices=indices,
            weights=np.ones(len(indices)),
            requested_rows=10,
            realized_rows=len(indices),
            budget_mode="native_realized",
            method_id="n02_coretab_xgb_subset",
            diagnostics={"upstream_commit": "locked123"},
        )

    monkeypatch.setattr(selector, "_run_binary", _binary)
    result = selector.select(X, y, 0.30)
    assert result.status == "success"
    assert result.requested_rows == 9
    assert result.realized_rows == 8
    assert result.budget_mode == "adapted_ova_realized"
    assert result.diagnostics["adapted"] is True
    assert result.diagnostics["native_equivalent"] is False
    assert [row["seed"] for row in result.diagnostics["class_runs"]] == [11, 12, 13]
    assert [row["kept_positive_rows"] for row in result.diagnostics["class_runs"]] == [3, 2, 3]
    assert all(y[index] in {0, 1, 2} for index in result.indices)


def test_coretab_ova_is_not_used_for_binary() -> None:
    selector = CoreTabXGBOVASelector(Path("unused"), seed=11)
    result = selector.select(np.ones((10, 2)), np.repeat([0, 1], 5), 0.2)
    assert result.status == "na_contract"
