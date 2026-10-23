"""Kiểm tra boundary AutoCoreset chỉ nhận đúng dòng thật từ train."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_autocoreset_native.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_autocoreset_native", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_roundtrip_accepts_upstream_dtype_promotion() -> None:
    module = _load_script()
    X = np.asarray([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32)
    y = np.asarray([-1, 1], dtype=np.int8)
    indices = module._map_rows_back(
        X, y, X[[1]].astype(np.float64), y[[1]].astype(np.float64),
    )
    assert indices.tolist() == [1]


def test_roundtrip_rejects_a_non_source_point() -> None:
    module = _load_script()
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.asarray([-1, 1], dtype=np.int8)
    with pytest.raises(RuntimeError, match="round-trip"):
        module._map_rows_back(
            X, y, np.asarray([[2.0, 3.0]]), np.asarray([1]),
        )

