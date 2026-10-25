"""Bổ sung giao thức CQ theo Review-2026-10-25 (dùng cùng các patch 12/10, 13/10, 22/10,
23/10).

1. a8_defined, a8_correct: hiệu chỉnh prior A8 về prior Pool B theo phase (QĐ30) và quy tắc
   tập huấn luyện thiếu hẳn một lớp.
2. ul_and_ul_a8: UL thô và UL_A8 của một lượt chạy từ xác suất từng dòng (QĐ25, QĐ30).
3. verdict_a8: kết luận "vượt", "hòa", "thua" trên UL_A8, kèm kết luận trên UL thô và cờ
   "phụ thuộc prior" (QĐ30). Quy tắc giống dev_verdict của bản 23/10.
4. phase_gap_report: B − A theo phase, B − S tại P4, gate S − A tại P4 và tín hiệu
   báo giảng viên (QĐ29).
5. e_class_report, family_bootstrap_ci: bộ chỉ số lớp E kèm CI theo họ (QĐ32).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

LABELS = (0, 1, 2)                      # E, G, A
PHASES = ("P1", "P2", "P3", "P4")


# 1. A8 ------------------------------------------------------------------------------------
def a8_defined(source_prior) -> bool:
    """A8 chỉ xác định khi tập huấn luyện có đủ ba lớp (prior hiệu dụng của mọi lớp > 0)."""
    s = np.asarray(source_prior, float)
    return bool(s.shape == (3,) and np.all(np.isfinite(s)) and np.all(s > 0))


def a8_correct(proba, source_prior, target_prior):
    """p~(y | x) tỷ lệ với p^(y | x) · pi_B,p(y) / pi_S(y), rồi chuẩn hóa lại theo y.

    source_prior: prior hiệu dụng (có trọng số, QĐ15) của tập huấn luyện của lượt chạy.
    target_prior: prior Pool B tại phase đang xét. Trả None khi A8 không xác định.
    Cùng công thức với prior_correct_probabilities của nhóm (training/course_metrics.py)."""
    if not a8_defined(source_prior):
        return None
    p = np.asarray(proba, float)
    q = p * (np.asarray(target_prior, float) / np.asarray(source_prior, float))
    return q / q.sum(axis=1, keepdims=True)


# 2. UL thô và UL_A8 ---------------------------------------------------------------------
def ul_and_ul_a8(proba_by_phase: dict, y_by_phase: dict, source_prior,
                 target_prior_by_phase: dict) -> dict:
    """Log loss từng phase trên toàn bộ lượt ghi danh của tập đánh giá, rồi trung bình P1–P4.
    ul_decision là đại lượng ra quyết định của QĐ30: UL_A8, hoặc UL thô khi A8 không xác định."""
    per_raw = {ph: float(log_loss(np.asarray(y_by_phase[ph]), np.asarray(proba_by_phase[ph]),
                                  labels=LABELS)) for ph in PHASES}
    out = {"ul_raw": float(np.mean(list(per_raw.values()))), "per_phase_raw": per_raw}
    if not a8_defined(source_prior):
        out.update(ul_a8=None, per_phase_a8=None, a8_undefined=True, ul_decision=out["ul_raw"])
        return out
    per_a8 = {}
    for ph in PHASES:
        corrected = a8_correct(proba_by_phase[ph], source_prior, target_prior_by_phase[ph])
        per_a8[ph] = float(log_loss(np.asarray(y_by_phase[ph]), corrected, labels=LABELS))
    ul_a8 = float(np.mean(list(per_a8.values())))
    out.update(ul_a8=ul_a8, per_phase_a8=per_a8, a8_undefined=False, ul_decision=ul_a8)
    return out


# 3. Kết luận trên dev ---------------------------------------------------------------------
def _verdict(ul_a_by_seed: dict, ul_b_by_seed, delta: float) -> dict:
    """Quy tắc QĐ25 (như dev_verdict, Review-2026-10-23, Phụ lục A): d_s = UL(A) − UL(B);
    "vượt" khi trung bình d <= −δ/2 và mọi d_s < 0; "thua" khi trung bình d >= δ/2 và mọi
    d_s > 0; còn lại "hòa". B có thể là một số (FullTrain-B, trung bình ba model seed)."""
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


def verdict_a8(ul_a8_a: dict, ul_a8_b, delta: float, ul_raw_a: dict | None = None,
               ul_raw_b=None) -> dict:
    """QĐ30: kết luận trên UL_A8 (truyền ul_decision của từng seed). Nếu có UL thô, tính thêm
    kết luận trên UL thô và gắn cờ prior_dependent khi hai kết luận khác nhau."""
    main = _verdict(ul_a8_a, ul_a8_b, delta)
    out = {"decision_quantity": "UL_A8", **main}
    if ul_raw_a is not None and ul_raw_b is not None:
        raw = _verdict(ul_raw_a, ul_raw_b, delta)
        out.update(raw_verdict=raw["verdict"], raw_mean_diff=raw["mean_diff"],
                   prior_dependent=raw["verdict"] != main["verdict"])
    return out


# 4. FullTrain-A, FullTrain-B và S (QĐ29) ---------------------------------------------------
def phase_gap_report(ll_a: dict, ll_b: dict, ll_s_p4: float | None = None,
                     gate_tol: float = 0.01) -> dict:
    """ll_a, ll_b: log loss từng phase của FullTrain-A và FullTrain-B (B: trung bình ba model
    seed); với các mô hình này UL_A8 = UL. Gate (bước 1 của D13): dừng khi S − A tại P4 lớn
    hơn gate_tol. B − S tại P4 là chi phí của việc học chung bốn phase, chỉ báo cáo mô tả.
    Tín hiệu báo giảng viên: UL(B) >= UL(A)."""
    per = {ph: {"A": float(ll_a[ph]), "B": float(ll_b[ph]),
                "B_minus_A": float(ll_b[ph] - ll_a[ph])} for ph in PHASES}
    ul_a = float(np.mean([ll_a[ph] for ph in PHASES]))
    ul_b = float(np.mean([ll_b[ph] for ph in PHASES]))
    out = {"per_phase": per, "ul_A": ul_a, "ul_B": ul_b, "ul_B_minus_A": ul_b - ul_a,
           "signal_B_not_better_than_A": bool(ul_b >= ul_a)}
    if ll_s_p4 is not None:
        s_minus_a = float(ll_s_p4 - ll_a["P4"])
        out.update(S_minus_A_P4=s_minus_a, B_minus_S_P4=float(ll_b["P4"] - ll_s_p4),
                   gate_stop=bool(s_minus_a > gate_tol))
    return out


# 5. Bộ chỉ số lớp E (QĐ32) -----------------------------------------------------------------
def _e_scores(p_e: np.ndarray, is_e: np.ndarray) -> dict:
    both = 0 < is_e.sum() < len(is_e)
    nll = -np.log(np.clip(p_e[is_e], 1e-15, 1.0)).mean() if is_e.any() else np.nan
    return {
        "nll_E": float(nll),
        "auc_E": float(roc_auc_score(is_e, p_e)) if both else float("nan"),
        "ap_E": float(average_precision_score(is_e, p_e)) if both else float("nan"),
    }


def e_class_report(proba, y, families, label: int = 0) -> dict:
    """Chỉ số lớp E trên một phase: NLL_E, AUC_E, AP_E (một-chọi-phần-còn-lại), recall và
    precision theo argmax, tỷ lệ argmax và xác suất trung bình so với tỷ lệ thực, số họ có E
    và số họ có ít nhất một E dự đoán đúng."""
    p = np.asarray(proba, float)
    y = np.asarray(y)
    fam = np.asarray(families).astype(str)
    is_e = y == label
    pred = p.argmax(axis=1)
    hit = (pred == label) & is_e
    n_pred = int((pred == label).sum())
    out = {
        "n_E": int(is_e.sum()),
        "n_families_with_E": int(np.unique(fam[is_e]).size),
        "n_families_with_TP_E": int(np.unique(fam[hit]).size),
        **_e_scores(p[:, label], is_e),
        "recall_E": float(hit.sum() / is_e.sum()) if is_e.any() else float("nan"),
        "precision_E": float(hit.sum() / n_pred) if n_pred else float("nan"),
        "argmax_share_E": float(n_pred / len(y)),
        "mean_proba_E": float(p[:, label].mean()),
        "observed_E": float(is_e.mean()),
    }
    return out


def family_bootstrap_ci(proba, y, families, n_bootstrap: int = 2000, seed: int = 20261025,
                        label: int = 0) -> dict:
    """CI 95% theo họ cho NLL_E, AUC_E, AP_E: mỗi lượt lấy lại họ có hoàn lại và ghép mọi dòng
    của các họ được chọn. Lượt không có E (hoặc chỉ có E) bị bỏ với AUC/AP; ghi số lượt hợp lệ."""
    p_e = np.asarray(proba, float)[:, label]
    is_e = np.asarray(y) == label
    keys, inverse = np.unique(np.asarray(families).astype(str), return_inverse=True)
    rows = [np.flatnonzero(inverse == k) for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in ("nll_E", "auc_E", "ap_E")}
    for _ in range(n_bootstrap):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(keys), len(keys))])
        scores = _e_scores(p_e[idx], is_e[idx])
        for m, v in scores.items():
            if np.isfinite(v):
                draws[m].append(v)
    point = _e_scores(p_e, is_e)
    return {m: {"point": point[m],
                "ci95": np.quantile(draws[m], [0.025, 0.975]).tolist() if draws[m] else None,
                "n_valid": len(draws[m]), "n_bootstrap": n_bootstrap,
                "bootstrap_unit": "course_family_id"} for m in draws}
