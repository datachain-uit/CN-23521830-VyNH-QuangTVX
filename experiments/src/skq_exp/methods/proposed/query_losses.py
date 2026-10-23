"""File này tạo đặc trưng loss ngoài-fold cho P04/P05 chỉ từ train.

Mỗi query learner được train trên các fold còn lại rồi dự đoán fold bị giữ ra.
Giá trị `-log p(y_i|x_i)` phản ánh vùng khó của learner mà không dùng dev/test,
sau đó được chuẩn hóa để ghép với biểu diễn kernel của SKQ.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def make_oof_query_losses(
    X: np.ndarray,
    y: np.ndarray,
    learner_ids: Iterable[str],
    *,
    seed: int,
    folds: int = 5,
    max_threads: int = 4,
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    labels, counts = np.unique(y, return_counts=True)
    n_splits = min(int(folds), int(counts.min()))
    if n_splits < 2:
        raise ValueError("Không đủ mẫu mỗi lớp để tạo OOF query loss")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    columns = []
    used = []
    for learner_id in learner_ids:
        prototype = _query_learner(learner_id, seed, max_threads)
        probability = np.zeros((len(y), len(labels)), dtype=np.float64)
        for train_idx, hold_idx in splitter.split(X, y):
            model = clone(prototype)
            model.fit(X[train_idx], y[train_idx])
            fold_prob = model.predict_proba(X[hold_idx])
            aligned = np.zeros((len(hold_idx), len(labels)), dtype=np.float64)
            for column, label in enumerate(model.classes_):
                aligned[:, int(np.flatnonzero(labels == label)[0])] = fold_prob[:, column]
            probability[hold_idx] = aligned
        true_columns = np.searchsorted(labels, y)
        loss = -np.log(np.clip(probability[np.arange(len(y)), true_columns], 1e-12, 1.0))
        columns.append(loss)
        used.append(learner_id)
    raw = np.column_stack(columns)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (raw - mean) / scale
    return standardized.astype(np.float32), {
        "query_learners": used,
        "oof_folds": n_splits,
        "loss": "negative log probability of true label",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _query_learner(learner_id: str, seed: int, max_threads: int):
    if learner_id == "lr":
        return LogisticRegression(max_iter=1500, solver="lbfgs", random_state=seed)
    if learner_id == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=max_threads
        )
    if learner_id == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("P05 cần xgboost cho query portfolio đã khóa") from error
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=max_threads, eval_metric="mlogloss",
        )
    raise KeyError(f"Query learner chưa đăng ký: {learner_id}")
