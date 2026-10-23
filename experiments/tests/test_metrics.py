"""Các test này kiểm tra metric dùng đúng prediction và giữ giá trị không xác định.

Chúng không kiểm lại công thức của scikit-learn mà kiểm hợp đồng đầu ra: đủ nhóm
metric, từng nhãn có confusion counts và probability thiếu không bị ghi thành 0.
"""

import numpy as np

from skq_exp.training.metrics import compute_all_metrics


def test_metrics_have_overall_and_per_label_views() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_prob = np.array([
        [0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2],
    ])
    result = compute_all_metrics(y_true, y_pred, y_prob, labels=np.array([0, 1, 2]))
    assert result["overall"]["f1_macro"] is not None
    assert result["overall"]["mcc"] is not None
    assert len(result["per_label"]) == 3
    assert {"tp", "fp", "tn", "fn", "f2"} <= set(result["per_label"][0])


def test_missing_probability_is_none_not_zero() -> None:
    result = compute_all_metrics(np.array([0, 1]), np.array([0, 1]), None)
    assert result["overall"]["roc_auc_macro"] is None
    assert result["overall"]["log_loss"] is None
    assert result["warnings"]

