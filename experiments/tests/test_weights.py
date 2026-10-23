"""Các test này kiểm tra method weight và class weight được ghép đúng cách.

Tỷ lệ tương đối giữa các trọng số phải được giữ, còn mean cuối cùng luôn bằng 1.
Weight 0 vẫn phải loại đóng góp của mẫu khỏi loss. Shape sai hoặc tổng bằng 0 bị
từ chối.
"""

import numpy as np
import pytest

from skq_exp.training.weights import (
    balanced_class_weights,
    effective_sample_weights,
    sample_weight_diagnostics,
)


def test_effective_weight_is_product_then_mean_normalized() -> None:
    y_full = np.array([0, 0, 0, 1])
    classes = balanced_class_weights(y_full)
    result = effective_sample_weights(
        np.array([0, 1]), np.array([2.0, 3.0]), classes
    )
    raw = np.asarray([2.0 * classes[0], 3.0 * classes[1]])
    assert np.allclose(result, raw / raw.mean())
    assert result.mean() == pytest.approx(1.0)


def test_global_scale_does_not_change_effective_weights() -> None:
    selected_y = np.array([0, 1, 1])
    original = effective_sample_weights(selected_y, np.array([1.0, 2.0, 4.0]), None)
    scaled = effective_sample_weights(selected_y, np.array([10.0, 20.0, 40.0]), None)
    assert np.allclose(original, scaled)


def test_weight_diagnostics_report_mean_one_without_raw_vector() -> None:
    weights = effective_sample_weights(
        np.array([0, 1]), np.array([1.0, 3.0]), None,
    )
    diagnostics = sample_weight_diagnostics(weights)
    assert diagnostics["mean"] == pytest.approx(1.0)
    assert diagnostics["sum"] == pytest.approx(2.0)
    assert diagnostics["count"] == 2


def test_invalid_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        effective_sample_weights(np.array([0, 1]), np.array([0.0, 0.0]), None)

