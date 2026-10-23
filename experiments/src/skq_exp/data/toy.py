"""File này tạo dữ liệu nhỏ để kiểm tra toàn bộ pipeline trong vài giây.

Dữ liệu chỉ dùng cho smoke test, không dùng trong bảng bài báo. Nó có ba lớp và
một parent structure đơn giản để kiểm tra cả selector global lẫn structured.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_toy_data(seed: int = 20261002) -> dict[str, np.ndarray]:
    X, y = make_classification(
        n_samples=2400, n_features=20, n_informative=12, n_redundant=4,
        n_classes=3, weights=[0.55, 0.30, 0.15], class_sep=1.2, random_state=seed,
    )
    row_ids = np.arange(len(y), dtype=np.int64)
    train, holdout = train_test_split(row_ids, test_size=0.30, stratify=y, random_state=seed)
    dev, test = train_test_split(holdout, test_size=0.50, stratify=y[holdout], random_state=seed)
    scaler = StandardScaler().fit(X[train])
    X_scaled = scaler.transform(X).astype(np.float32)
    parent = (X_scaled[:, 0] > np.median(X_scaled[train, 0])).astype(np.int64)
    return {
        "X_train": X_scaled[train], "y_train": y[train], "row_ids_train": train,
        "X_dev": X_scaled[dev], "y_dev": y[dev], "row_ids_dev": dev,
        "X_test": X_scaled[test], "y_test": y[test], "row_ids_test": test,
        "parent_train": parent[train],
    }

