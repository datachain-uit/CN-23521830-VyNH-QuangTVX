"""Kiểm thử cq_patch_1025.py (Review-2026-10-25)."""
import numpy as np
import pytest
from scipy.special import logsumexp

try:                                    # trong repo: experiments/src/skq_exp/data/
    from skq_exp.data.cq_patch_1025 import (PHASES, a8_correct, a8_defined, e_class_report,
                                            family_bootstrap_ci, phase_gap_report,
                                            ul_and_ul_a8, verdict_a8)
except ImportError:                     # chạy độc lập, tệp đặt cùng thư mục
    from cq_patch_1025 import (PHASES, a8_correct, a8_defined, e_class_report,
                               family_bootstrap_ci, phase_gap_report, ul_and_ul_a8, verdict_a8)

PI_B = np.array([522, 36006, 95378]) / 131906           # Pool B (train V4)
PI_DEV = np.array([263, 6606, 15848]) / 22717           # dev V4, mỗi phase
E_TEST = 263 / 23975
PI_TEST = np.array([E_TEST, 0.176, 1 - E_TEST - 0.176])  # test V4 (G, A làm tròn)
MU = np.array([[2.4, 1.0, 0, 0, 0, 0], [0.9, 0.9, 0.3, 0, 0, 0], [0, 0, 0, 0, 0, 0]])


def _posterior(X, pi):
    L = -0.5 * ((X[:, None, :] - MU[None]) ** 2).sum(-1) + np.log(pi)
    return np.exp(L - logsumexp(L, axis=1, keepdims=True))


def _sample(pi, n, seed):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=pi)
    return MU[y] + rng.standard_normal((n, 6)), y


def test_a8_is_identity_when_training_prior_equals_pool_b():
    p = np.random.default_rng(0).dirichlet([1, 3, 6], size=200)
    assert np.allclose(a8_correct(p, PI_B, PI_B), p, atol=1e-12)


def test_a8_undefined_when_a_class_is_missing():
    assert not a8_defined([0.0, 0.3, 0.7])
    p = np.full((4, 3), 1 / 3)
    y = {ph: np.array([0, 1, 2, 2]) for ph in PHASES}
    out = ul_and_ul_a8({ph: p for ph in PHASES}, y, [0.0, 0.3, 0.7], {ph: PI_B for ph in PHASES})
    assert out["a8_undefined"] and out["ul_a8"] is None
    assert out["ul_decision"] == out["ul_raw"]


def test_same_prior_shift_loses_on_dev_wins_on_test_raw_but_ties_after_a8():
    """O29: tập huấn luyện có tỷ lệ G bằng 0,8 lần Pool B, cùng thông tin. Theo UL thô nó
    'thua' FullTrain-B trên dev nhưng 'vượt' trên test; theo UL_A8 là 'hòa' ở cả hai."""
    pi_s = PI_B * np.array([1, 0.8, 1])
    pi_s /= pi_s.sum()
    target = {ph: PI_B for ph in PHASES}
    for pi_eval, raw_expected in ((PI_DEV, "thua"), (PI_TEST, "vượt")):
        raw_core, dec_core, full = {}, {}, {}
        for seed in (11, 29, 47):
            X, y = _sample(pi_eval, 200_000, seed)
            y_ph = {ph: y for ph in PHASES}
            b = ul_and_ul_a8({ph: _posterior(X, PI_B) for ph in PHASES}, y_ph, PI_B, target)
            c = ul_and_ul_a8({ph: _posterior(X, pi_s) for ph in PHASES}, y_ph, pi_s, target)
            assert b["ul_a8"] == pytest.approx(b["ul_raw"], abs=1e-12)    # A8 không đổi B
            raw_core[seed], dec_core[seed], full[seed] = c["ul_raw"], c["ul_decision"], b["ul_raw"]
        out = verdict_a8(dec_core, full, 0.008, ul_raw_a=raw_core, ul_raw_b=full)
        assert out["raw_verdict"] == raw_expected
        assert out["verdict"] == "hòa" and out["prior_dependent"]
        assert abs(out["mean_diff"]) < 1e-9


def test_verdict_rule_matches_qd25():
    assert verdict_a8({11: 0.50, 29: 0.49, 47: 0.52}, 0.51, 0.007)["verdict"] == "hòa"
    assert verdict_a8({11: 0.50, 29: 0.50, 47: 0.50}, 0.51, 0.007)["verdict"] == "vượt"
    assert verdict_a8({11: 0.52, 29: 0.52, 47: 0.52}, 0.51, 0.007)["verdict"] == "thua"
    with pytest.raises(ValueError):
        verdict_a8({11: 0.5}, {29: 0.5}, 0.007)


def test_phase_gap_report_reproduces_24_10_numbers_and_gate():
    a = {"P1": 0.808234, "P2": 0.520065, "P3": 0.469778, "P4": 0.434546}
    b = {"P1": 0.545926, "P2": 0.503529, "P3": 0.482561, "P4": 0.473985}
    r = phase_gap_report(a, b, ll_s_p4=0.44)          # LL_P4(S) = 0,44 chỉ để minh họa
    assert r["ul_A"] == pytest.approx(0.558156, abs=1e-6)
    assert r["ul_B"] == pytest.approx(0.501500, abs=1e-6)
    assert r["per_phase"]["P1"]["B_minus_A"] == pytest.approx(-0.262308, abs=1e-6)
    assert r["per_phase"]["P4"]["B_minus_A"] == pytest.approx(0.039439, abs=1e-6)
    assert not r["signal_B_not_better_than_A"]
    assert not r["gate_stop"]                                    # S − A = 0,005454 <= 0,01
    assert r["B_minus_S_P4"] == pytest.approx(0.033985, abs=1e-6)
    assert phase_gap_report(a, b, ll_s_p4=0.446)["gate_stop"]    # S − A = 0,011454 > 0,01


def test_e_class_report_counts_families_and_rates():
    y = np.array([0, 0, 0, 1, 2, 2, 2, 2])
    fam = np.array(["f1", "f1", "f2", "f1", "f2", "f3", "f3", "f3"])
    p = np.array([[.6, .2, .2], [.1, .3, .6], [.5, .3, .2], [.2, .5, .3],
                  [.7, .1, .2], [.1, .2, .7], [.1, .1, .8], [.2, .2, .6]])
    r = e_class_report(p, y, fam)
    assert (r["n_E"], r["n_families_with_E"], r["n_families_with_TP_E"]) == (3, 2, 2)
    assert r["recall_E"] == pytest.approx(2 / 3) and r["precision_E"] == pytest.approx(2 / 3)
    assert r["argmax_share_E"] == pytest.approx(3 / 8)
    assert r["nll_E"] == pytest.approx(-np.mean(np.log([.6, .1, .5])))


def test_family_bootstrap_ci_brackets_point_estimate():
    X, y = _sample(PI_DEV, 20_000, 5)
    fam = np.random.default_rng(1).integers(0, 60, len(y))
    ci = family_bootstrap_ci(_posterior(X, PI_B), y, fam, n_bootstrap=300)
    for m in ("nll_E", "auc_E", "ap_E"):
        lo, hi = ci[m]["ci95"]
        assert lo <= ci[m]["point"] <= hi and ci[m]["n_valid"] == 300


def test_a8_matches_group_implementation():
    """Chạy trong repo của nhóm: hai cài đặt A8 phải trùng nhau."""
    cm = pytest.importorskip("skq_exp.training.course_metrics")
    p = np.random.default_rng(3).dirichlet([1, 3, 6], size=50)
    src = {"E": 0.008, "G": 0.270, "A": 0.722}
    tgt = {"E": 0.004, "G": 0.273, "A": 0.723}
    ours = a8_correct(p, [src[k] for k in "EGA"], [tgt[k] for k in "EGA"])
    assert np.allclose(ours, cm.prior_correct_probabilities(p, src, tgt), atol=1e-12)
