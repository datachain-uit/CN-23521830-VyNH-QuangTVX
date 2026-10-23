"""Bổ sung giao thức CQ theo Review-2026-10-23 (dùng cùng các patch 12/10, 13/10, 22/10).

1. ul_log_loss: UL chính mới, log loss đa lớp trung bình P1–P4 (QĐ25).
2. skill_margin: biên δ_LL bằng 5% mức cải thiện của FullTrain-B so với dự đoán bằng
   prior (QĐ25).
3. headroom_rho_ll: quy tắc ρ* của QĐ12 viết lại cho đại lượng nhỏ là tốt (QĐ26).
4. dev_verdict: "vượt", "hòa", "thua" trên dev theo QĐ5 và QĐ20 với UL mới (QĐ25).
5. select_learner_view: quy tắc chọn cấu hình learner của chẩn đoán D13 (QĐ24).
6. code_gain_share, subgroup_log_loss, calibration_in_the_large, course_contributions:
   các kiểm tra cơ chế của D13.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

LABELS = [0, 1, 2]                      # E, G, A
PHASES = ("P1", "P2", "P3", "P4")


# 1. UL = log loss trung bình P1–P4 ----------------------------------------------------
def ul_log_loss(proba_by_phase: dict, y_by_phase: dict) -> dict:
    """Mỗi phase tính log loss trên toàn bộ lượt ghi danh của tập đánh giá, rồi lấy trung
    bình bốn phase. Giữ giá trị từng phase vì K3 dùng thêm P1. Nhỏ hơn là tốt hơn."""
    per = {ph: float(log_loss(np.asarray(y_by_phase[ph]), np.asarray(proba_by_phase[ph]),
                              labels=LABELS)) for ph in PHASES}
    return {"per_phase": per, "ul": float(np.mean(list(per.values())))}


def prior_log_loss(y_eval: np.ndarray, train_prior: np.ndarray) -> float:
    """Log loss của dự đoán hằng bằng prior train (mức sàn D05)."""
    y_eval = np.asarray(y_eval)
    p = np.tile(np.asarray(train_prior, float), (len(y_eval), 1))
    return float(log_loss(y_eval, p, labels=LABELS))


# 2. Biên non-inferiority theo mức cải thiện ------------------------------------------
def skill_margin(ll_prior: float, ll_full: float, fraction: float = 0.05,
                 decimals: int = 3) -> float:
    """δ_LL = fraction × (LL_prior − LL_FullTrain-B), cả hai là UL trên dev với cấu hình
    learner đã chốt. Tính một lần tại điểm kiểm tra 29/10, làm tròn rồi đóng băng."""
    skill = ll_prior - ll_full
    if not skill > 0:
        raise ValueError("FullTrain-B không tốt hơn dự đoán bằng prior: chưa đặt được biên")
    return max(round(fraction * skill, decimals), 10.0 ** -decimals)


# 3. Headroom với log loss -----------------------------------------------------------
def headroom_rho_ll(ll_full: float, ll_c02: dict, gap: float,
                    grid=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """ll_c02: {ρ: {seed: UL}}. ρ* là mức lớn nhất có trung bình qua seed của
    UL(C02, ρ) − UL(FullTrain-B) ≥ gap (gap = 2·δ_LL); không mức nào đạt thì ρ* là mức
    nhỏ nhất (fallback của QĐ12)."""
    rows = {}
    for rho in grid:
        v = np.asarray(list(ll_c02[rho].values()), float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        delta = float(v.mean() - ll_full)
        rows[rho] = {"mean": float(v.mean()), "sd": sd, "se": sd / math.sqrt(len(v)),
                     "delta": delta, "eligible": bool(delta >= gap)}
    ok = [r for r in grid if rows[r]["eligible"]]
    return {"rows": rows, "rho_star": max(ok) if ok else min(grid), "fallback": not ok,
            "gap": gap}


# 4. Kết luận trên dev cho đại lượng nhỏ là tốt -------------------------------------------
def dev_verdict(ul_a_by_seed: dict, ul_b_by_seed: dict | float, delta: float) -> dict:
    """d_s = UL(A) − UL(B) theo seed. "vượt": trung bình d ≤ −δ/2 và mọi seed d_s < 0
    (cùng chiều, QĐ5); "thua": trung bình d ≥ δ/2 và mọi seed d_s > 0; còn lại "hòa"
    (QĐ20: hòa thì chọn phương pháp rẻ hơn). B có thể là một số (FullTrain-B)."""
    seeds = sorted(ul_a_by_seed)
    b = {s: ul_b_by_seed for s in seeds} if np.isscalar(ul_b_by_seed) else ul_b_by_seed
    if set(seeds) != set(b):
        raise ValueError("hai phương pháp phải có cùng danh sách seed")
    d = np.array([ul_a_by_seed[s] - b[s] for s in seeds], float)
    if d.mean() <= -delta / 2 and (d < 0).all():
        verdict = "vượt"
    elif d.mean() >= delta / 2 and (d > 0).all():
        verdict = "thua"
    else:
        verdict = "hòa"
    return {"mean_diff": float(d.mean()), "diffs": d.tolist(), "verdict": verdict}


# 5. Chọn cấu hình learner (D13) --------------------------------------------------------
def select_learner_view(ul_by_config: dict, order: list, tol: float = 0.002) -> str:
    """Chọn cấu hình có UL dev thấp nhất. Các cấu hình cách mức thấp nhất không quá tol
    coi là ngang nhau; khi đó chọn cấu hình đứng trước trong order (gần đặc tả đã đăng ký
    nhất)."""
    missing = set(ul_by_config) - set(order)
    if missing:
        raise ValueError(f"cấu hình không có trong thứ tự ưu tiên: {sorted(missing)}")
    best = min(ul_by_config.values())
    return next(c for c in order if c in ul_by_config and ul_by_config[c] <= best + tol)


# 6. Kiểm tra cơ chế ------------------------------------------------------------------
def code_gain_share(booster, feature_names: list, code_features: list) -> float:
    """Tỷ trọng total_gain của các cột mã trong một mô hình XGBoost đã huấn luyện."""
    score = booster.get_score(importance_type="total_gain")
    names = booster.feature_names or [f"f{j}" for j in range(len(feature_names))]
    to_name = dict(zip(names, feature_names))
    total = float(sum(score.values()))
    codes = set(code_features)
    part = sum(v for k, v in score.items() if to_name.get(k, k) in codes)
    return part / total if total > 0 else 0.0


def subgroup_log_loss(proba: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Log loss theo nhóm, ví dụ người học hoặc tập giảng viên đã gặp ở train hay chưa."""
    proba, y, groups = np.asarray(proba), np.asarray(y), np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        ll = float(log_loss(y[m], proba[m], labels=LABELS))
        out[str(g)] = {"n": int(m.sum()), "log_loss": ll}
    return out


def calibration_in_the_large(proba: np.ndarray, y: np.ndarray) -> dict:
    """Trung bình xác suất dự đoán, tỷ lệ thực và tỷ lệ argmax của từng lớp."""
    proba, y = np.asarray(proba, float), np.asarray(y)
    pred = proba.argmax(1)
    return {lab: {"mean_proba": float(proba[:, k].mean()), "observed": float((y == k).mean()),
                  "argmax_share": float((pred == k).mean())} for k, lab in enumerate("EGA")}


def row_log_loss(proba: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Log loss của từng dòng, −log p̂_y(x)."""
    y = np.asarray(y)
    return -np.log(np.clip(np.asarray(proba, float)[np.arange(len(y)), y], eps, 1.0))


def course_contributions(ll_rows_a: np.ndarray, ll_rows_b: np.ndarray, course_ids: np.ndarray,
                         top: int = 3) -> dict:
    """Phân rã chênh lệch log loss trung bình (A − B) theo khóa học. Đóng góp của khóa c là
    tổng (ℓ_A,i − ℓ_B,i) trên các dòng của c chia cho N; tổng các đóng góp bằng chênh lệch
    toàn tập. Dùng để biết chênh lệch có tập trung ở vài khóa lớn hay không."""
    d = np.asarray(ll_rows_a, float) - np.asarray(ll_rows_b, float)
    s = pd.Series(d).groupby(np.asarray(course_ids)).sum() / len(d)
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    head = s.head(top)
    total = float(d.mean())
    return {"total": total, "top": {str(k): float(v) for k, v in head.items()},
            "top_share": float(head.sum() / total) if total != 0 else None}
