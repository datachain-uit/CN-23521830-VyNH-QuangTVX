"""File này tạo các downstream learner với cấu hình đã khóa.

LR, RF, XGBoost, CatBoost và MLP đều nhận sample weight thật. Mỗi model chỉ được
tune trên FullTrain train/inner-validation; hàm ở đây nhận cấu hình thắng đã freeze
và không nhìn test để thay đổi tham số.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LearnerSpec:
    learner_id: str
    display_name: str
    role: str
    mandatory: bool
    supports_weight: bool


LEARNER_SPECS = {
    spec.learner_id: spec for spec in [
        LearnerSpec("lr", "Logistic/Softmax Regression", "linear sanity", True, True),
        LearnerSpec("rf", "Random Forest", "bagging transfer", True, True),
        LearnerSpec("xgb", "XGBoost", "confirmatory primary", True, True),
        LearnerSpec("cat", "CatBoost", "boosting transfer", True, True),
        LearnerSpec("mlp", "Weighted MLP", "neural transfer", True, True),
        LearnerSpec("tabnet", "TabNet", "appendix", False, False),
        LearnerSpec("tabm", "TabM", "appendix", False, False),
    ]
}


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "lr": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
    "rf": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1},
    "xgb": {
        "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "n_estimators": 1000,
    },
    "cat": {"depth": 6, "learning_rate": 0.05, "iterations": 1000},
    "mlp": {
        "hidden_sizes": (256, 128), "learning_rate": 1e-3,
        "weight_decay": 0.0, "dropout": 0.1, "batch_size": 2048,
        "max_epochs": 100, "patience": 10,
    },
}


def build_learner(
    learner_id: str,
    *,
    seed: int = 42,
    n_classes: int | None = None,
    input_dim: int | None = None,
    params: dict[str, Any] | None = None,
    max_threads: int = 4,
):
    if learner_id not in LEARNER_SPECS:
        raise KeyError(f"Learner chưa đăng ký: {learner_id}")
    merged = {**DEFAULT_PARAMS.get(learner_id, {}), **(params or {})}
    if learner_id == "lr":
        standardize = bool(merged.pop("standardize", False))
        if standardize:
            return StandardizedLogisticRegression(seed=seed, params=merged)
        return LogisticRegression(random_state=seed, **merged)
    if learner_id == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=max_threads, **merged)
    if learner_id == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("Cần cài extra boosting để chạy XGBoost") from error
        return XGBClassifier(
            random_state=seed, n_jobs=max_threads, eval_metric="mlogloss", **merged
        )
    if learner_id == "cat":
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise RuntimeError("Cần cài extra boosting để chạy CatBoost") from error
        return CatBoostClassifier(
            random_seed=seed, verbose=False, thread_count=max_threads, **merged
        )
    if learner_id == "mlp":
        if input_dim is None or n_classes is None:
            raise ValueError("MLP cần input_dim và n_classes")
        return WeightedMLPClassifier(
            input_dim, n_classes, seed=seed, max_threads=max_threads, **merged
        )
    raise RuntimeError(f"{learner_id} là appendix/resource-gated và chưa được mở")


class StandardizedLogisticRegression:
    """LR với scaler fit train-only; giữ API/cost fields mà evaluator cần.

    Wrapper này không nhìn dev/test khi fit scaler. ``standardize`` là một phần
    của learner config được lưu vào protocol hash, không phải tiền xử lý ngầm.
    """

    def __init__(self, *, seed: int, params: dict[str, Any]) -> None:
        self.seed = seed
        self.params = dict(params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "StandardizedLogisticRegression":
        self.scaler_ = StandardScaler()
        transformed = self.scaler_.fit_transform(np.asarray(X))
        self.model_ = LogisticRegression(random_state=self.seed, **self.params)
        self.model_.fit(transformed, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_iter_ = self.model_.n_iter_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.scaler_.transform(np.asarray(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self.scaler_.transform(np.asarray(X)))


class WeightedMLPClassifier:
    """MLP nhỏ dùng weighted loss theo từng mẫu và early stopping trên dev."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        *,
        hidden_sizes: tuple[int, ...] = (256, 128),
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.1,
        batch_size: int = 2048,
        max_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
        max_threads: int = 4,
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.max_threads = max_threads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "WeightedMLPClassifier":
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as error:
            raise RuntimeError("Cần cài extra deep để chạy MLP") from error
        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, self.max_threads))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers: list[nn.Module] = []
        width = self.input_dim
        for hidden in self.hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(self.dropout)])
            width = hidden
        layers.append(nn.Linear(width, self.n_classes))
        self.model_ = nn.Sequential(*layers).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        weights = np.ones(len(y), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
            torch.as_tensor(weights, dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        best_state = None
        best_loss = float("inf")
        stale = 0
        for _epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device_), yb.to(self.device_), wb.to(self.device_)
                optimizer.zero_grad()
                losses = loss_fn(self.model_(xb), yb)
                loss = torch.sum(losses * wb) / torch.clamp(torch.sum(wb), min=1e-12)
                loss.backward()
                optimizer.step()
            if eval_set is not None:
                validation_loss = self._validation_loss(eval_set, loss_fn)
                if validation_loss < best_loss - 1e-8:
                    best_loss = validation_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.classes_ = np.arange(self.n_classes)
        return self

    def _validation_loss(self, eval_set, loss_fn) -> float:
        import torch
        X_val, y_val = eval_set
        self.model_.eval()
        with torch.no_grad():
            # `X_val` có thể là read-only memmap; torch.tensor tạo bản sao an toàn.
            logits = self.model_(torch.tensor(np.asarray(X_val), dtype=torch.float32, device=self.device_))
            losses = loss_fn(logits, torch.as_tensor(y_val, dtype=torch.long, device=self.device_))
        return float(losses.mean().item())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.tensor(
                    np.asarray(X[start:start + self.batch_size]),
                    dtype=torch.float32, device=self.device_,
                )
                outputs.append(torch.softmax(self.model_(batch), dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

