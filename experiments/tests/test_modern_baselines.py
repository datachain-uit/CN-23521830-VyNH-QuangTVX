"""Toy contract cho bốn baseline mới, không tải Internet trong CI."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from skq_exp.methods.adapted import modern_baselines as module
from skq_exp.methods.adapted.modern_baselines import (
    CCSOOFXGBSelector,
    DataOOBOfficialSelector,
    KernelThinningClasswiseSelector,
    ModerateDSTabularSelector,
)


def _toy(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    y = np.repeat(np.arange(3), 40)
    X[:, 0] += y
    return X, y


def test_goodpoints_runtime_provenance_requires_locked_commit(monkeypatch) -> None:
    class FakeDistribution:
        version = "0.6.3"

        @staticmethod
        def read_text(name: str) -> str | None:
            assert name == "direct_url.json"
            return (
                '{"url":"https://github.com/microsoft/goodpoints.git",'
                '"vcs_info":{"vcs":"git","commit_id":"locked-commit"}}'
            )

    monkeypatch.setattr(module.metadata, "distribution", lambda _: FakeDistribution())
    provenance = module._verify_installed_goodpoints("locked-commit")
    assert provenance["commit"] == "locked-commit"
    with pytest.raises(RuntimeError, match="cần 'other-commit'"):
        module._verify_installed_goodpoints("other-commit")


def test_moderate_ds_adapted_is_exact_and_declared(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "verify_locked_repo", lambda _: "moderate-commit")
    X, y = _toy()
    result = ModerateDSTabularSelector(tmp_path, 11).select(X, y, 0.10)
    assert result.status == "success"
    assert result.realized_rows == 12
    assert result.diagnostics["adapted"] is True
    assert result.diagnostics["native_equivalent"] is False
    assert result.diagnostics["upstream_commit"] == "moderate-commit"


def test_ccs_uses_oof_scores_and_upstream_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "verify_locked_repo", lambda _: "ccs-commit")
    X, y = _toy()
    selector = CCSOOFXGBSelector(tmp_path, 11)
    monkeypatch.setattr(selector, "_oof_el2n", lambda _X, _y: (np.linspace(0, 1, len(_y)), 5))
    monkeypatch.setattr(
        selector, "_upstream_stratified",
        lambda scores, requested: np.arange(requested, dtype=np.int64),
    )
    result = selector.select(X, y, 0.10)
    assert result.status == "success"
    assert result.realized_rows == 12
    assert result.diagnostics["difficulty_score"] == "OOF_XGBoost_EL2N"
    assert result.diagnostics["oof_folds"] == 5
    assert result.diagnostics["adapted"] is True


def test_dataoob_keeps_highest_values_without_proxy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "verify_locked_repo", lambda _: "dataoob-commit")
    X, y = _toy()

    class FakeOfficialRF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, fit_X, fit_y):
            assert len(fit_X) == len(fit_y)
            return self

        def evaluate_oob_accuracy(self, fit_X, fit_y):
            del fit_y
            return pd.Series(np.arange(len(fit_X), dtype=np.float64))

    selector = DataOOBOfficialSelector(tmp_path, 11, {"n_trees": 50})
    monkeypatch.setattr(selector, "_load_upstream_class", lambda: FakeOfficialRF)
    result = selector.select(X, y, 0.10)
    assert result.status == "success"
    assert result.indices.tolist() == list(range(119, 107, -1))
    assert result.diagnostics["selection_rule"] == "highest_OOB_value"
    assert result.diagnostics["native_equivalent"] is False
    assert result.diagnostics["value_estimator_equivalent"] is True
    assert "top-k" in result.diagnostics["declared_adaptation"]


def test_kernel_thinning_classwise_exact_budget_and_patch_log(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "verify_locked_repo", lambda _: "goodpoints-commit")
    monkeypatch.setattr(
        module,
        "_verify_installed_goodpoints",
        lambda commit: {
            "distribution": "goodpoints",
            "version": "test",
            "source_url": "https://github.com/microsoft/goodpoints.git",
            "commit": commit,
        },
    )
    X, y = _toy()
    package = types.ModuleType("goodpoints")
    kt_module = types.ModuleType("goodpoints.kt")
    compress_module = types.ModuleType("goodpoints.compress")

    def thin(points, rounds, *args, **kwargs):
        del args, kwargs
        step = 2 ** rounds
        return np.arange(0, len(points), step, dtype=np.int64)

    kt_module.thin = thin
    compress_module.compresspp_kt = lambda points, *args, **kwargs: np.arange(
        max(1, int(np.sqrt(len(points)))), dtype=np.int64,
    )
    package.kt = kt_module
    package.compress = compress_module
    monkeypatch.setitem(sys.modules, "goodpoints", package)
    monkeypatch.setitem(sys.modules, "goodpoints.kt", kt_module)
    monkeypatch.setitem(sys.modules, "goodpoints.compress", compress_module)
    result = KernelThinningClasswiseSelector(tmp_path, 11).select(X, y, 0.10)
    assert result.status == "success"
    assert result.realized_rows == 12
    assert len(np.unique(y[result.indices])) == 3
    assert result.diagnostics["upstream_commit"] == "goodpoints-commit"
    assert result.diagnostics["runtime_package_provenance"]["commit"] == "goodpoints-commit"
    assert result.diagnostics["adapted"] is True
    assert result.budget_mode == "exact_classwise_compresspp"
    assert all(run["compresspp_candidate_rows"] >= run["quota"] for run in result.diagnostics["class_runs"])


def test_kernel_thinning_does_not_import_unbuilt_checkout(monkeypatch, tmp_path) -> None:
    """Checkout audit không được che package Cython đã build trong site-packages."""
    monkeypatch.setattr(module, "verify_locked_repo", lambda _: "goodpoints-commit")
    monkeypatch.setattr(
        module,
        "_verify_installed_goodpoints",
        lambda commit: {
            "distribution": "goodpoints",
            "version": "test",
            "source_url": "https://github.com/microsoft/goodpoints.git",
            "commit": commit,
        },
    )
    monkeypatch.setattr(
        module,
        "prepend_sys_path",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("Không được prepend checkout GoodPoints chưa build")
        ),
    )
    X, y = _toy()
    package = types.ModuleType("goodpoints")
    kt_module = types.ModuleType("goodpoints.kt")
    compress_module = types.ModuleType("goodpoints.compress")
    kt_module.thin = lambda points, rounds, *args, **kwargs: np.arange(
        0, len(points), 2 ** rounds, dtype=np.int64,
    )
    package.kt = kt_module
    compress_module.compresspp_kt = lambda points, *args, **kwargs: np.arange(
        max(1, int(np.sqrt(len(points)))), dtype=np.int64,
    )
    package.compress = compress_module
    monkeypatch.setitem(sys.modules, "goodpoints", package)
    monkeypatch.setitem(sys.modules, "goodpoints.kt", kt_module)
    monkeypatch.setitem(sys.modules, "goodpoints.compress", compress_module)
    result = KernelThinningClasswiseSelector(tmp_path, 11).select(X, y, 0.10)
    assert result.status == "success"
