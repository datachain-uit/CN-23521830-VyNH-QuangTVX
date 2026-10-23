"""File này chọn các dòng có trung bình RFF gần với trung bình của nhóm.

Mỗi vòng chọn điểm làm phần dư trung bình giảm nhiều nhất. Khi nhiều điểm có cùng
điểm số, một hash từ seed và row ID quyết định thứ tự để kết quả tái lập được.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _tie_value(seed: int, row_id: int) -> int:
    raw = f"{seed}:{int(row_id)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def kernel_herding(
    Z: np.ndarray, n_select: int, row_ids: np.ndarray, seed: int, check=None,
) -> np.ndarray:
    if n_select <= 0 or n_select > len(Z):
        raise ValueError("n_select phải nằm trong [1, số dòng của nhóm]")
    target = np.asarray(Z, dtype=np.float64).mean(axis=0)
    matrix = np.asarray(Z, dtype=np.float64)
    squared_norms = np.einsum("ij,ij->i", matrix, matrix)
    running = np.zeros(Z.shape[1], dtype=np.float64)
    available = np.ones(len(Z), dtype=bool)
    selected: list[int] = []
    tie_order = np.asarray([_tie_value(seed, row_id) for row_id in row_ids], dtype=np.uint64)
    for step in range(n_select):
        if check is not None and step % 8 == 0:
            check("herding")
        desired = (step + 1) * target - running
        # argmin ||running + z_j - (t+1)mu||^2 tương đương argmax biểu thức này.
        # Hạng norm là bắt buộc vì RFF row norm không hoàn toàn không đổi.
        scores = matrix @ desired - 0.5 * squared_norms
        scores[~available] = -np.inf
        best = np.max(scores)
        candidates = np.flatnonzero(np.isclose(scores, best, rtol=1e-12, atol=1e-12))
        winner = int(candidates[np.argmin(tie_order[candidates])])
        selected.append(winner)
        available[winner] = False
        running += Z[winner]
    return np.asarray(selected, dtype=np.int64)

