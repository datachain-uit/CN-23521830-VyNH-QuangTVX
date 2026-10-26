"""Kiểm thử cq_patch_1026.py (Review-2026-10-26). Chạy: pytest -q test_cq_patch_1026.py"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
 
try:  # trong repo của nhóm
    from skq_exp.data.cq_patch_1026 import (
        PANEL_SEEDS, verdict_panel, verdict_power, phase_source_priors, ul_a8_by_phase,
        support_report, e_reference_floor, e_run_flags, course_constant_columns,
        quasi_id_audit, constant_marginal_log_loss)
except ImportError:  # chạy độc lập
    from cq_patch_1026 import (
        PANEL_SEEDS, verdict_panel, verdict_power, phase_source_priors, ul_a8_by_phase,
        support_report, e_reference_floor, e_run_flags, course_constant_columns,
        quasi_id_audit, constant_marginal_log_loss)
 
DELTA = 0.010                                   # δ_LL đề xuất của nhóm (QĐ33)
PHASES = ("P1", "P2", "P3", "P4")
 
 
def _seeds(values):
    return dict(zip(PANEL_SEEDS, values))
 
 
# 1. Quy tắc kết luận 5 seed -----------------------------------------------------------------
def test_verdict_panel_rules():
    base = _seeds([0.505] * 5)
    win5 = _seeds([0.495] * 5)
    assert verdict_panel(win5, base, DELTA)["verdict"] == "vượt"
    win4 = _seeds([0.495, 0.494, 0.496, 0.493, 0.506])          # 4/5 tốt hơn, mean −0,0082
    assert verdict_panel(win4, base, DELTA)["verdict"] == "vượt"
    assert verdict_panel(win4, base, DELTA, k_required=5)["verdict"] == "hòa"
    win3 = _seeds([0.475, 0.475, 0.475, 0.506, 0.506])          # mean −0,0176 nhưng chỉ 3/5
    assert verdict_panel(win3, base, DELTA)["verdict"] == "hòa"
    small = _seeds([0.501] * 5)                                  # 5/5 nhưng |mean| < δ/2
    assert verdict_panel(small, base, DELTA)["verdict"] == "hòa"
    lose = _seeds([0.515, 0.512, 0.516, 0.511, 0.504])
    assert verdict_panel(lose, base, DELTA)["verdict"] == "thua"
    assert verdict_panel(win5, 0.505, DELTA)["verdict"] == "vượt"   # B là một số
    with pytest.raises(ValueError):
        verdict_panel({11: 0.49, 29: 0.49, 47: 0.49}, {11: 0.5, 29: 0.5, 47: 0.5}, DELTA)
 
 
def test_verdict_panel_prior_dependent_flag():
    base = _seeds([0.505] * 5)
    out = verdict_panel(_seeds([0.495] * 5), base, DELTA,
                        ul_raw_a=_seeds([0.503] * 5), ul_raw_b=base)
    assert out["verdict"] == "vượt" and out["raw_verdict"] == "hòa" and out["prior_dependent"]
 
 
# 2. Xác suất "vượt" -------------------------------------------------------------------------
def test_verdict_power_reproduces_appendix_table():
    sd = 0.010441                                                # SD của C02 tại 0,5%
    kw = dict(sd_a=sd, sd_b=sd, delta=DELTA, n_sim=100_000)
    assert 0.10 < verdict_power(0.0, n_seeds=3, k_required=3, **kw) < 0.13
    assert 0.12 < verdict_power(0.0, n_seeds=5, k_required=4, **kw) < 0.16
    assert 0.39 < verdict_power(0.01, n_seeds=3, k_required=3, **kw) < 0.44
    assert 0.58 < verdict_power(0.01, n_seeds=5, k_required=4, **kw) < 0.63
    rates = [verdict_power(g, n_seeds=5, k_required=4, **kw) for g in (0, .005, .01, .02)]
    assert rates == sorted(rates)
 
 
# 3. A8 theo phase ------------------------------------------------------------------------------
def _phase_shift_case(seed=0, n=20_000):
    """Pool B có prior như nhau ở mọi phase; tập huấn luyện có prior khác nhau theo phase.
    Mô hình "đúng" theo tập huấn luyện có posterior ∝ L(x|y)·pi_S,p(y)."""
    rng = np.random.default_rng(seed)
    pi_b = np.array([0.004, 0.273, 0.723])
    pi_s = {"P1": [0.001, 0.25, 0.749], "P2": [0.003, 0.27, 0.727],
            "P3": [0.006, 0.28, 0.714], "P4": [0.010, 0.30, 0.690]}
    proba, ytrue = {}, {}
    for ph in PHASES:
        lik = rng.dirichlet([0.6, 0.6, 0.6], size=n)            # L(x|y) của mỗi dòng
        post_b = lik * pi_b
        post_b /= post_b.sum(1, keepdims=True)
        ytrue[ph] = np.array([rng.choice(3, p=p) for p in post_b])
        q = lik * np.asarray(pi_s[ph])
        proba[ph] = q / q.sum(1, keepdims=True)
        proba[ph + "_true"] = post_b
    return proba, ytrue, pi_b, pi_s
 
 
def test_a8_phase_recovers_pool_b_posterior_when_learner_learns_phase_priors():
    proba, y, pi_b, pi_s = _phase_shift_case()
    g = np.mean([pi_s[ph] for ph in PHASES], axis=0)
    priors = {"global": g, "by_phase": {ph: np.asarray(pi_s[ph]) for ph in PHASES}}
    out = ul_a8_by_phase({ph: proba[ph] for ph in PHASES}, y, priors, {ph: pi_b for ph in PHASES})
    truth = np.mean([log_loss(y[ph], proba[ph + "_true"], labels=[0, 1, 2]) for ph in PHASES])
    assert out["a8_phase_basis"] == "phase" and out["decision_basis"] == "A8_global"
    assert out["ul_a8_phase"] == pytest.approx(truth, abs=1e-12)
    assert out["ul_decision"] == out["ul_a8_global"] > out["ul_a8_phase"]
 
 
def test_a8_phase_equals_global_when_phase_priors_equal_and_undefined_cases():
    proba, y, pi_b, _ = _phase_shift_case(seed=1, n=2_000)
    p = {ph: proba[ph] for ph in PHASES}
    s = np.array([0.003, 0.26, 0.737])
    same = ul_a8_by_phase(p, y, {"global": s, "by_phase": {ph: s for ph in PHASES}},
                          {ph: pi_b for ph in PHASES})
    assert same["ul_a8_phase"] == pytest.approx(same["ul_a8_global"], abs=1e-12)
    miss_cell = {"global": s, "by_phase": {**{ph: s for ph in PHASES},
                                           "P1": np.array([0.0, 0.3, 0.7])}}
    out = ul_a8_by_phase(p, y, miss_cell, {ph: pi_b for ph in PHASES})
    assert out["ul_a8_phase"] is None and out["a8_phase_basis"].startswith("undefined")
    assert out["ul_decision"] == pytest.approx(same["ul_a8_global"])
    raw = ul_a8_by_phase(p, y, {"global": np.array([0.0, 0.3, 0.7]), "by_phase": None},
                         {ph: pi_b for ph in PHASES})
    assert raw["decision_basis"] == "raw_missing_class" and raw["ul_decision"] == raw["ul_raw"]
 
 
def test_verdict_panel_a8_dependent_flag():
    base = _seeds([0.505] * 5)
    out = verdict_panel(_seeds([0.497] * 5), base, DELTA,
                        ul_phase_a=_seeds([0.502] * 5), ul_phase_b=base)
    assert out["verdict"] == "vượt" and out["a8_phase_verdict"] == "hòa" and out["a8_dependent"]
 
 
def test_phase_source_priors_weighted():
    y = np.array([0, 1, 2, 2, 1, 2])
    ph = np.array(["P1", "P1", "P1", "P2", "P2", "P2"])
    pr = phase_source_priors(y, ph, weights=[2, 1, 1, 1, 1, 2])
    assert np.allclose(pr["by_phase"]["P1"], [0.5, 0.25, 0.25])
    assert np.allclose(pr["by_phase"]["P2"], [0.0, 0.25, 0.75])
    assert np.isnan(pr["by_phase"]["P3"]).all()
 
 
# 4. Support của SKQ-PC ----------------------------------------------------------------------------
def test_support_report_matches_smoke_counts():
    rng = np.random.default_rng(11)
    n, n_zero = 2638, 284                                        # smoke SKQ-PC seed 11, ρ = 0,5%
    w = rng.uniform(50, 400, n)
    w[rng.choice(n, n_zero, replace=False)] = 0.0
    y = rng.choice(3, n, p=[0.004, 0.273, 0.723])
    ph = rng.choice(PHASES, n)
    rep = support_report(w, y, ph)
    assert rep["budget_rows_physical"] == 2638 and rep["rows_positive_weight"] == 2354
    assert rep["zero_weight_share"] == pytest.approx(284 / 2638)
    assert 0 < rep["normalized_ess"] < 1 and len(rep["strata"]) == 12
    with pytest.raises(ValueError):
        support_report(-w, y, ph)
 
 
# 5. Lớp E -----------------------------------------------------------------------------------------
def test_e_reference_floor_and_flags():
    prev = 263 / 22717                                           # tỷ lệ E trên dev
    ok = e_reference_floor(0.091486, prev)                       # FullTrain-B seed 42
    assert ok["pass"] and ok["ratio"] == pytest.approx(7.90, abs=0.01)
    assert not e_reference_floor(0.02, prev)["pass"]
    flags = e_run_flags({"ap_E": 0.016, "observed_E": prev, "mean_proba_E": 0.004,
                         "n_families_with_TP_E": 0}, n_e_rows_selected=10, n_e_rows_positive=0)
    assert flags["flag_E_absent"] and flags["flag_E_ranking_below_floor"]
    assert flags["flag_E_no_family_with_TP"]
 
 
# 6. Ngữ cảnh tĩnh và quasi-ID ---------------------------------------------------------------------
def test_course_constant_columns_and_quasi_id_audit():
    nan = np.nan
    train = pd.DataFrame({                           # dòng đầu của c1 không có sự kiện video
        "course_id": ["c1", "c1", "c2", "c2", "c3"],
        "course_family_id": ["f1", "f1", "f2", "f2", "f3"],
        "teacher_count": [2, 2, 1, 1, 2], "course_video_count": [nan, 10, 30, 30, 10],
        "segment_count": [nan, 5, 1, 2, 4], "always_one": [1, 1, 1, 1, 1]})
    dev = pd.DataFrame({
        "course_id": ["d1", "d2"], "course_family_id": ["g1", "g2"],
        "teacher_count": [2, 3], "course_video_count": [10, 40],
        "segment_count": [1, 1], "always_one": [1, 1]})
    groups = course_constant_columns(train, ["teacher_count", "course_video_count",
                                             "segment_count", "always_one"])
    assert groups["constant"] == ["teacher_count"]
    assert groups["constant_when_observed"] == ["course_video_count"]
    cols = groups["g_ctx"]
    rep = quasi_id_audit(train, dev, cols)
    assert rep["train_courses"] == 3 and rep["train_distinct_vectors"] == 2
    assert rep["train_share_unique"] == pytest.approx(1 / 3)
    assert rep["dev_courses_matching_train"] == 1
    assert rep["dev_matches"][0] == {"dev_course": "d1", "dev_family": "g1",
                                     "train_families": ["f1", "f3"]}
 
 
# 7. Phân bố biên --------------------------------------------------------------------------------
def test_constant_marginal_log_loss():
    y = np.array([0, 1, 2, 2, 2, 1])
    p = np.tile([0.02, 0.30, 0.68], (6, 1))
    out = constant_marginal_log_loss(p, y)
    assert out["ll_model"] == pytest.approx(out["ll_constant_marginal"])
    assert np.allclose(out["marginal"], [0.02, 0.30, 0.68])
