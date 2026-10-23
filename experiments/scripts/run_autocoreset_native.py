"""Script này chạy AutoCoreset trong boundary riêng và xuất index/weight chuẩn.

Nó chỉ đọc train đã xử lý, gọi đúng hàm upstream ở commit khóa và ghi artifact để
runner chính nhập lại. Compatibility patch chỉ xử lý API NumPy/sklearn đã đổi;
mọi patch và commit đều được ghi trong manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from skq_exp.artifacts import atomic_json, sha256_file, stable_hash  # noqa: E402
from skq_exp.config import ExperimentConfig  # noqa: E402
from skq_exp.methods.native.common import verify_locked_repo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    processed = config.paths.processed_root / args.dataset
    repo = config.paths.external_root / "autocoreset"
    locked_commit = verify_locked_repo(repo)
    output = config.paths.artifact_root / "native" / "autocoreset" / args.dataset / f"ss{args.seed}"
    X = np.load(processed / "X_train.npy")
    y = np.load(processed / "y_train.npy")
    requested = max(1, int(round(len(y) * config.budget_ratio)))
    fingerprint = stable_hash({
        "X_train": sha256_file(processed / "X_train.npy"),
        "y_train": sha256_file(processed / "y_train.npy"),
    })
    started = time.perf_counter()
    np.random.seed(args.seed)
    _install_compatibility_aliases()
    sys.path.insert(0, str(repo))
    try:
        import coreset_utils
        import main as upstream
        problem = "binary_logistic_regression" if len(np.unique(y)) == 2 else "multiclass_logistic_regression"
        y_native = np.where(y == np.unique(y)[0], -1, 1) if problem.startswith("binary") else _onehot(y)
        loss, model_f = coreset_utils.obtainLossAndModel(problem, fit_intercept=True, C=1)
        chunks, size_per_class = coreset_utils.chunkize(
            np.arange(len(X)), y, requested, solver=problem, fair=True
        )
        _, _, chunk_idxs, size_per_chunk = coreset_utils.shuffleDataViaChunks(X, y_native, chunks)
        upstream.OPT_VAL = None
        upstream.OPTIMALITY_COUNTER = 0
        extra = {
            "randomized_sol": "uniform", "replace": False, "maintain_classes": True,
            "old_weights": np.ones(len(X)), "chunk_idxs": chunk_idxs,
            "size_per_chunk": size_per_chunk, "labels": y_native,
            "size_per_class": size_per_class,
            "VSC": upstream.VSC.AutoCoreset_CARATHEODORY,
        }
        C, y_c, weights = upstream.initiateAutoCore(
            X, y_native, loss, requested, upstream.INITIAL_RANDOM_SOLS,
            upstream.patienceBasedOptimalityCriteon, model_f, **extra,
        )
    finally:
        sys.path.pop(0)
    indices = _map_rows_back(X, y_native, np.asarray(C), np.asarray(y_c))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        len(indices) != len(weights) or len(np.unique(indices)) != len(indices)
        or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0
    ):
        raise RuntimeError("AutoCoreset output vi phạm unique-index/weight contract")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "indices.npy", indices)
    np.save(output / "weights.npy", weights)
    atomic_json(output / "manifest.json", {
        "method_id": "n_autocoreset_native",
        "upstream_repo": "https://github.com/alaamaalouf/AutoCoreset.git",
        "upstream_commit": locked_commit,
        "dataset_id": args.dataset,
        "dataset_fingerprint": fingerprint,
        "selector_seed": args.seed,
        "requested_rows": requested,
        "realized_rows": int(len(indices)),
        "compatibility_patches": [
            "numpy.infty alias",
            "sklearn OneHotEncoder sparse->sparse_output adapter",
            "dtype-normalized exact row identity mapping for upstream subset output",
        ],
        "timings": {"total": time.perf_counter() - started},
    })
    print(output)


def _install_compatibility_aliases() -> None:
    if not hasattr(np, "infty"):
        np.infty = np.inf  # type: ignore[attr-defined]
    import sklearn.preprocessing
    original = sklearn.preprocessing.OneHotEncoder

    class CompatOneHotEncoder(original):
        def __init__(self, *, sparse=None, **kwargs):
            if sparse is not None:
                kwargs["sparse_output"] = sparse
            super().__init__(**kwargs)

    sklearn.preprocessing.OneHotEncoder = CompatOneHotEncoder


def _onehot(y: np.ndarray) -> np.ndarray:
    labels, encoded = np.unique(y, return_inverse=True)
    result = np.zeros((len(y), len(labels)))
    result[np.arange(len(y)), encoded] = 1
    return result


def _map_rows_back(X, y, C, y_c):
    """Ánh xạ subset upstream về index mà không phụ thuộc dtype trung gian.

    AutoCoreset chọn ``C = P[C_prime]`` nên đây vẫn là subset dòng thật. Một số
    nhánh sklearn/upstream nâng float32/int thành float64; so sánh byte trực tiếp
    khi đó thất bại dù giá trị và dòng không đổi. Ép đầu ra về đúng dtype nguồn
    trước khi lập khóa giữ nguyên identity dòng, không dùng nearest-neighbour và
    không biến điểm tổng hợp thành subset.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    C = np.asarray(C)
    y_c = np.asarray(y_c)
    if C.ndim != X.ndim or C.shape[1:] != X.shape[1:]:
        raise RuntimeError(
            f"AutoCoreset trả shape feature không khớp train: {C.shape} != (*, {X.shape[1:]})"
        )
    if y_c.shape[1:] != y.shape[1:]:
        raise RuntimeError(
            f"AutoCoreset trả shape label không khớp train: {y_c.shape} != (*, {y.shape[1:]})"
        )

    def key(row, label):
        row_bytes = np.ascontiguousarray(row, dtype=X.dtype).tobytes()
        label_bytes = np.ascontiguousarray(label, dtype=y.dtype).tobytes()
        return row_bytes, label_bytes

    buckets: dict[tuple[bytes, bytes], list[int]] = {}
    for index, (row, label) in enumerate(zip(X, y)):
        buckets.setdefault(key(row, label), []).append(index)
    mapped = []
    for row, label in zip(C, y_c):
        row_key = key(row, label)
        if not buckets.get(row_key):
            raise RuntimeError("Không thể round-trip một dòng AutoCoreset về train index")
        mapped.append(buckets[row_key].pop(0))
    return np.asarray(mapped, dtype=np.int64)


if __name__ == "__main__":
    main()
