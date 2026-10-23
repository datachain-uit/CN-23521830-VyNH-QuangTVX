"""Chọn rho* trên dev bằng FullTrain-B và C02 theo QĐ12.

Input phải có đúng các mức 0.5/1/2/5% và tập selector seed đã đăng ký. Hàm không
nhận metric SKQ/baseline khác, do đó không thể vô tình dùng selector để chọn budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np


HEADROOM_RATIOS = (0.005, 0.01, 0.02, 0.05)
HEADROOM_SEEDS = (11, 29, 47)
V4_HEADROOM_SEEDS = (11, 29, 47, 71, 101)
HEADROOM_GAP = 0.01


def choose_headroom_budget(
    *,
    full_train_b_ul: float,
    c02_dev_ul: dict[float, dict[int, float]],
    gap: float = HEADROOM_GAP,
    required_seeds: tuple[int, ...] = HEADROOM_SEEDS,
) -> dict[str, Any]:
    """Trả decision có toàn bộ input; không đọc test và không nhận method khác."""
    if not np.isfinite(full_train_b_ul) or not 0 <= full_train_b_ul <= 1:
        raise ValueError("full_train_b_ul phải hữu hạn trong [0,1]")
    if gap <= 0:
        raise ValueError("headroom gap phải dương")
    if set(c02_dev_ul) != set(HEADROOM_RATIOS):
        raise ValueError(f"C02 phải có đúng rho={HEADROOM_RATIOS}")
    normalized: dict[str, Any] = {}
    eligible = []
    for ratio in HEADROOM_RATIOS:
        seed_values = c02_dev_ul[ratio]
        if set(seed_values) != set(required_seeds):
            raise ValueError(
                f"rho={ratio} phải có đúng seed={required_seeds}, không thiếu/trùng"
            )
        values = np.asarray([seed_values[seed] for seed in required_seeds], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"UL C02 không hợp lệ ở rho={ratio}")
        mean = float(values.mean())
        delta = float(full_train_b_ul - mean)
        if delta >= gap:
            eligible.append(ratio)
        normalized[str(ratio)] = {
            "seed_values": {str(seed): float(seed_values[seed]) for seed in required_seeds},
            "mean": mean,
            "delta_full_minus_c02": delta,
            "eligible": delta >= gap,
        }
    selected = max(eligible) if eligible else min(HEADROOM_RATIOS)
    return {
        "schema_version": 1,
        "evidence_split": "dev",
        "allowed_inputs": ["FullTrain-B", "C02"],
        "selector_seeds": list(required_seeds),
        "full_train_b_ul": float(full_train_b_ul),
        "gap": float(gap),
        "ratios": normalized,
        "rho_star": float(selected),
        "fallback_to_smallest": not bool(eligible),
        "secondary_descriptive_ratio": 0.05 if selected != 0.05 else None,
    }
