"""Bổ sung giao thức CQ theo Review-2026-10-26 (dùng cùng cq_patch_1025.py).
 
1. PANEL_SEEDS, verdict_panel: kết luận "vượt", "hòa", "thua" của panel dev với 5 selector
   seed (QĐ33); cờ "phụ thuộc prior" như verdict_a8 (QĐ30).
2. verdict_power: mô phỏng xác suất kết luận "vượt" của quy tắc (Phụ lục A.1).
3. phase_source_priors, ul_a8_by_phase: UL_A8 toàn tập (QĐ30, đại lượng quyết định) và UL_A8
   theo prior từng phase của tập huấn luyện (phân tích độ nhạy, O32).
4. support_report: số dòng vật lý, số dòng trọng số dương, ESS và trọng số 0 theo tầng
   phase × lớp (QĐ33).
5. e_reference_floor, e_run_flags: sàn kiểm tra lớp E của FullTrain-B và cờ mô tả cho mọi
   lượt chạy (QĐ34).
6. course_constant_columns, quasi_id_audit: nhóm ngữ cảnh tĩnh cấp khóa và kiểm tra
   quasi-identifier (QĐ35).
7. constant_marginal_log_loss: log loss của dự đoán hằng bằng phân bố biên của mô hình (Y35).
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
 
LABELS = (0, 1, 2)                      # E, G, A
PHASES = ("P1", "P2", "P3", "P4")
PANEL_SEEDS = (11, 29, 47, 71, 101)     # trùng 5 selector seed của headroom C02
 
 
# 1. Kết luận của panel dev (QĐ33) ----------------------------------------------------------
def _verdict_k(ul_a: dict, ul_b, delta: float, k_required: int) -> dict:
    """d_s = UL(A, s) − UL(B, s), ghép theo selector seed. "vượt": trung bình d <= −δ/2 và ít
    nhất k_required seed có d_s < 0; "thua": đối xứng; còn lại "hòa". B có thể là một số
    (FullTrain-B, trung bình ba model seed)."""
    seeds = sorted(ul_a)
    if set(seeds) != set(PANEL_SEEDS):
        raise ValueError(f"panel dev cần đúng các selector seed {PANEL_SEEDS}")
    b = {s: ul_b for s in seeds} if np.isscalar(ul_b) else ul_b
    if set(b) != set(seeds):
        raise ValueError("hai phương pháp phải có cùng danh sách seed")
    if not (np.isfinite(delta) and delta > 0):
        raise ValueError("δ_LL phải dương")
    d = np.array([ul_a[s] - b[s] for s in seeds], float)
    if not np.isfinite(d).all():
        raise ValueError("giá trị UL không hợp lệ")
    n_neg, n_pos = int((d < 0).sum()), int((d > 0).sum())
    if d.mean() <= -delta / 2 and n_neg >= k_required:
        verdict = "vượt"
    elif d.mean() >= delta / 2 and n_pos >= k_required:
        verdict = "thua"
    else:
        verdict = "hòa"
    return {"mean_diff": float(d.mean()), "diffs": d.tolist(), "n_seeds_better": n_neg,
            "n_seeds_worse": n_pos, "k_required": k_required, "verdict": verdict}
 
 
def verdict_panel(ul_dec_a: dict, ul_dec_b, delta: float, ul_raw_a: dict | None = None,
                  ul_raw_b=None, ul_phase_a: dict | None = None, ul_phase_b=None,
                  k_required: int = 4) -> dict:
    """QĐ33 + QĐ30: kết luận trên ul_decision (UL_A8 toàn tập, hoặc UL thô khi A8 không xác
    định) với 5 seed và ngưỡng ≥ 4/5. Nếu có UL thô: thêm kết luận thô và cờ prior_dependent.
    Nếu có UL_A8 theo phase (O32): thêm kết luận theo phase và cờ a8_dependent."""
    main = _verdict_k(ul_dec_a, ul_dec_b, delta, k_required)
    out = {"decision_quantity": "UL_A8", "rule": f">={k_required}/{len(PANEL_SEEDS)}", **main}
    if ul_raw_a is not None and ul_raw_b is not None:
        raw = _verdict_k(ul_raw_a, ul_raw_b, delta, k_required)
        out.update(raw_verdict=raw["verdict"], raw_mean_diff=raw["mean_diff"],
                   prior_dependent=raw["verdict"] != main["verdict"])
    if ul_phase_a is not None and ul_phase_b is not None:
        ph = _verdict_k(ul_phase_a, ul_phase_b, delta, k_required)
        out.update(a8_phase_verdict=ph["verdict"], a8_phase_mean_diff=ph["mean_diff"],
                   a8_dependent=ph["verdict"] != main["verdict"])
    return out
 
 
# 2. Xác suất "vượt" của quy tắc --------------------------------------------------------------
def verdict_power(true_gain: float, *, sd_a: float, sd_b: float, delta: float, n_seeds: int,
                  k_required: int, n_sim: int = 200_000, seed: int = 20261026) -> float:
    """Tỷ lệ lượt mô phỏng kết luận "vượt" khi UL(A) thấp hơn UL(B) đúng true_gain. Giả định:
    giá trị theo seed của hai phương pháp độc lập, phân phối chuẩn với SD sd_a, sd_b."""
    rng = np.random.default_rng(seed)
    d = rng.normal(-true_gain, np.hypot(sd_a, sd_b), size=(n_sim, n_seeds))
    return float(((d.mean(axis=1) <= -delta / 2) & ((d < 0).sum(axis=1) >= k_required)).mean())
 
 
# 3. A8 theo phase (O32) -------------------------------------------------------------------
def _prior(y, w) -> np.ndarray:
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    mass = np.array([w[y == k].sum() for k in LABELS], float)
    return mass / mass.sum() if mass.sum() > 0 else np.full(3, np.nan)
 
 
def phase_source_priors(y, phase, weights=None) -> dict:
    """Prior hiệu dụng (có trọng số đã chuẩn hóa theo QĐ15) của tập huấn luyện: toàn tập và
    theo từng phase. Phase không có dòng nào cho prior NaN."""
    y, phase = np.asarray(y), np.asarray(phase)
    w = None if weights is None else np.asarray(weights, float)
    by_phase = {ph: _prior(y[phase == ph], None if w is None else w[phase == ph]) for ph in PHASES}
    return {"global": _prior(y, w), "by_phase": by_phase}
 
 
def _ok(p) -> bool:
    p = np.asarray(p, float)
    return bool(p.shape == (3,) and np.isfinite(p).all() and (p > 0).all())
 
 
def _correct(proba, source, target):
    q = np.asarray(proba, float) * (np.asarray(target, float) / np.asarray(source, float))
    return q / q.sum(axis=1, keepdims=True)
 
 
def _ul_corrected(proba_by_phase, y_by_phase, source_by_phase, target_prior_by_phase) -> dict:
    per = {ph: float(log_loss(y_by_phase[ph], _correct(proba_by_phase[ph], source_by_phase[ph],
                                                       target_prior_by_phase[ph]),
                              labels=LABELS)) for ph in PHASES}
    return {"ul": float(np.mean(list(per.values()))), "per_phase": per}
 
 
def ul_a8_by_phase(proba_by_phase: dict, y_by_phase: dict, priors: dict,
                   target_prior_by_phase: dict) -> dict:
    """Hai cách hiệu chỉnh A8 về prior Pool B theo phase.
    - Toàn tập (QĐ30, đại lượng quyết định): p~ tỷ lệ với p^ · pi_B,p / pi_S, với pi_S là prior
      hiệu dụng của toàn tập huấn luyện. Thiếu lớp: A8 không xác định, dùng UL thô.
    - Theo phase (độ nhạy, O32): p~ tỷ lệ với p^ · pi_B,p / pi_S,p, với pi_S,p là prior của các
      dòng phase p trong tập huấn luyện; đúng khi learner học được prior riêng của từng phase
      (có phase_fraction). Không xác định khi một phase thiếu dòng hoặc thiếu một lớp."""
    raw = {ph: float(log_loss(y_by_phase[ph], proba_by_phase[ph], labels=LABELS)) for ph in PHASES}
    out = {"ul_raw": float(np.mean(list(raw.values()))), "per_phase_raw": raw,
           "ul_a8_global": None, "ul_a8_phase": None}
    g = priors["global"]
    if _ok(g):
        r = _ul_corrected(proba_by_phase, y_by_phase, {ph: g for ph in PHASES}, target_prior_by_phase)
        out.update(ul_a8_global=r["ul"], per_phase_a8_global=r["per_phase"],
                   ul_decision=r["ul"], decision_basis="A8_global")
    else:
        out.update(ul_decision=out["ul_raw"], decision_basis="raw_missing_class")
    by_phase = priors.get("by_phase") or {}
    if all(_ok(by_phase.get(ph, [np.nan])) for ph in PHASES):
        r = _ul_corrected(proba_by_phase, y_by_phase, by_phase, target_prior_by_phase)
        out.update(ul_a8_phase=r["ul"], per_phase_a8_phase=r["per_phase"], a8_phase_basis="phase")
    else:
        out["a8_phase_basis"] = "undefined_empty_phase_class_cell"
    return out
 
 
# 4. Ngân sách và support (QĐ33) ---------------------------------------------------------
def support_report(weights, y, phase) -> dict:
    """Ngân sách tính theo dòng vật lý (exact_total). Báo thêm support dương, tỷ lệ trọng số 0,
    ESS chuẩn hóa (Σw)² / (n·Σw²) và theo tầng phase × lớp: số dòng, số dòng dương,
    max(w)/mean(w) trên các dòng dương."""
    w = np.asarray(weights, float)
    y, phase = np.asarray(y), np.asarray(phase)
    if not (len(w) == len(y) == len(phase)) or (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("trọng số phải hữu hạn, không âm và khớp số dòng")
    pos = w > 0
    out = {"budget_rows_physical": int(len(w)), "rows_positive_weight": int(pos.sum()),
           "zero_weight_share": float(1 - pos.mean()),
           "normalized_ess": float(w.sum() ** 2 / (len(w) * (w ** 2).sum())), "strata": {}}
    for ph in PHASES:
        for k, lab in zip(LABELS, "EGA"):
            m = (phase == ph) & (y == k)
            wp = w[m & pos]
            out["strata"][f"{ph}_{lab}"] = {
                "rows": int(m.sum()), "rows_positive": int(len(wp)),
                "max_over_mean_weight": float(wp.max() / wp.mean()) if len(wp) else None}
    return out
 
 
# 5. Lớp E (QĐ34) --------------------------------------------------------------------------
def e_reference_floor(ap_e: float, prevalence_e: float, factor: float = 2.0) -> dict:
    """Sàn kiểm tra cho FullTrain-B (từng model seed, E gộp P1–P4 trên dev):
    AP_E >= factor × tỷ lệ E. Không đạt thì dừng và báo giảng viên trước panel."""
    ratio = float(ap_e / prevalence_e)
    return {"ap_E": float(ap_e), "prevalence_E": float(prevalence_e), "ratio": ratio,
            "factor": factor, "pass": bool(ratio >= factor)}
 
 
def e_run_flags(e_pooled: dict, n_e_rows_selected: int, n_e_rows_positive: int,
                factor: float = 2.0) -> dict:
    """Cờ mô tả cho một lượt coreset (không là điều kiện loại): e_pooled là kết quả
    e_class_report gộp P1–P4 (bản 25/10)."""
    return {
        "E_rows_selected": int(n_e_rows_selected), "E_rows_positive_weight": int(n_e_rows_positive),
        "flag_E_absent": bool(n_e_rows_positive == 0),
        "flag_E_ranking_below_floor": bool(e_pooled["ap_E"] < factor * e_pooled["observed_E"]),
        "flag_E_no_family_with_TP": bool(e_pooled["n_families_with_TP_E"] == 0),
        "mean_p_over_observed_E": float(e_pooled["mean_proba_E"] / e_pooled["observed_E"]),
    }
 
 
# 6. Ngữ cảnh tĩnh cấp khóa (QĐ35) -------------------------------------------------------------
def course_constant_columns(frame: pd.DataFrame, candidates, course_col: str = "course_id") -> dict:
    """Dùng trên raw_features của Pool B train. Trả hai nhóm cột không hằng trên toàn tập:
    - "constant": đúng một giá trị (kể cả NULL) trong mỗi khóa (ví dụ teacher_count);
    - "constant_when_observed": trong mỗi khóa, mọi giá trị không thiếu đều bằng nhau, nhưng có
      dòng NULL (ví dụ course_video_count chỉ có ở dòng có sự kiện video trước mốc cắt).
    G_ctx của QĐ35 là hợp của hai nhóm."""
    cols = [c for c in candidates if c in frame]
    g = frame.groupby(course_col, sort=False)[cols]
    with_null = g.nunique(dropna=False).max()
    observed = g.nunique(dropna=True).max()
    overall = frame[cols].nunique(dropna=True)
    const = [c for c in cols if with_null[c] == 1 and overall[c] > 1]
    when_obs = [c for c in cols if c not in const and observed[c] <= 1 and overall[c] > 1]
    return {"constant": const, "constant_when_observed": when_obs, "g_ctx": const + when_obs}
 
 
def _keys(frame: pd.DataFrame, cols, decimals: int) -> list:
    vals = frame[list(cols)].astype(float).round(decimals).fillna(-1e300).to_numpy()
    return [tuple(row) for row in vals]
 
 
def _per_course(frame: pd.DataFrame, cols, course_col: str, family_col: str) -> pd.DataFrame:
    """Một dòng mỗi khóa; mỗi cột lấy giá trị không thiếu đầu tiên của khóa (groupby.first)."""
    return frame.groupby(course_col, sort=True)[[family_col, *cols]].first().reset_index()
 
 
def quasi_id_audit(train: pd.DataFrame, dev: pd.DataFrame, cols, course_col: str = "course_id",
                   family_col: str = "course_family_id", decimals: int = 9) -> dict:
    """Vector ngữ cảnh tĩnh của mỗi khóa (một dòng mỗi khóa, dùng raw_features). Báo: số khóa,
    số vector khác nhau và tỷ lệ khóa có vector duy nhất ở train; số khóa dev có vector trùng
    khít một khóa train và các cặp (khóa dev, họ dev, các họ train trùng) để đối chiếu với
    quy tắc họ của QĐ19."""
    tr = _per_course(train, cols, course_col, family_col)
    dv = _per_course(dev, cols, course_col, family_col)
    ktr, kdv = _keys(tr, cols, decimals), _keys(dv, cols, decimals)
    counts: dict = {}
    fam_by_key: dict = {}
    for k, fam in zip(ktr, tr[family_col].astype(str)):
        counts[k] = counts.get(k, 0) + 1
        fam_by_key.setdefault(k, set()).add(fam)
    matches = [{"dev_course": str(c), "dev_family": str(f), "train_families": sorted(fam_by_key[k])}
               for c, f, k in zip(dv[course_col], dv[family_col], kdv) if k in fam_by_key]
    return {"n_cols": len(cols), "train_courses": int(len(tr)),
            "train_distinct_vectors": int(len(counts)),
            "train_share_unique": float(np.mean([counts[k] == 1 for k in ktr])),
            "dev_courses": int(len(dv)), "dev_courses_matching_train": len(matches),
            "dev_matches": matches}
 
 
# 7. Phân bố biên của dự đoán (Y35) ---------------------------------------------------------------
def constant_marginal_log_loss(proba, y) -> dict:
    """Log loss của dự đoán hằng bằng trung bình xác suất của mô hình trên chính tập đánh giá.
    Nếu gần bằng log loss của mô hình thì phần thấp hơn UL_prior đến từ phân bố biên (ví dụ
    dừng sớm trên dev kéo xác suất về gần prior dev), không từ thứ hạng."""
    p = np.asarray(proba, float)
    y = np.asarray(y)
    m = p.mean(axis=0)
    return {"marginal": m.tolist(),
            "ll_model": float(log_loss(y, p, labels=LABELS)),
            "ll_constant_marginal": float(log_loss(y, np.tile(m, (len(y), 1)), labels=LABELS))}
