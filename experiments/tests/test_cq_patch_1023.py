"""Original Appendix A tests from Review 2026-10-23; only import path adapted."""
import numpy as np
import pytest
from sklearn.metrics import f1_score, log_loss, recall_score, roc_auc_score

from skq_exp.data.cq_patch_1023 import (calibration_in_the_large, code_gain_share, course_contributions,
                           dev_verdict, headroom_rho_ll, prior_log_loss, row_log_loss,
                           select_learner_view, skill_margin, subgroup_log_loss, ul_log_loss)

TRAIN_PRIOR = np.array([522, 36006, 95378]) / 131906          # train V4 (data card)
Y_DEV = np.repeat([0, 1, 2], [263, 6606, 15848])                 # dev V4 mỗi phase


def test_prior_baseline_reproduces_d05_and_ul_matches_sklearn():
    assert prior_log_loss(Y_DEV, TRAIN_PRIOR) == pytest.approx(0.667817, abs=2e-6)
    rng = np.random.default_rng(0)
    p = {ph: rng.dirichlet([1, 3, 6], size=len(Y_DEV)) for ph in ("P1", "P2", "P3", "P4")}
    y = {ph: Y_DEV for ph in p}
    out = ul_log_loss(p, y)
    expected = np.mean([log_loss(Y_DEV, p[ph], labels=[0, 1, 2]) for ph in p])
    assert out["ul"] == pytest.approx(expected)
    assert set(out["per_phase"]) == {"P1", "P2", "P3", "P4"}


def test_skill_margin_on_current_numbers():
    assert skill_margin(0.667817, 0.536384) == pytest.approx(0.007)   # FullTrain-B hiện tại
    with pytest.raises(ValueError):
        skill_margin(0.60, 0.61)


def test_headroom_rule_and_fallback():
    seeds = (11, 29, 47, 71, 101)
    c02 = {r: {s: m + 0.001 * (i - 2) for i, s in enumerate(seeds)}
           for r, m in {0.005: 0.560, 0.01: 0.535, 0.02: 0.515, 0.05: 0.506}.items()}
    out = headroom_rho_ll(0.500, c02, gap=0.014)
    assert out["rho_star"] == 0.02 and not out["fallback"]
    assert out["rows"][0.05]["eligible"] is False
    flat = {r: {s: 0.505 for s in seeds} for r in c02}
    assert headroom_rho_ll(0.500, flat, gap=0.014)["rho_star"] == 0.005


def test_dev_verdict_needs_size_and_all_seeds():
    ref = {11: 0.520, 29: 0.522, 47: 0.521}
    verdict = lambda a, b: dev_verdict(a, b, delta=0.01)["verdict"]
    assert verdict({11: 0.512, 29: 0.514, 47: 0.515}, ref) == "vượt"
    assert verdict({11: 0.518, 29: 0.519, 47: 0.520}, ref) == "hòa"     # chênh < δ/2
    assert verdict({11: 0.505, 29: 0.524, 47: 0.506}, ref) == "hòa"     # không đủ 3/3 seed
    assert verdict({11: 0.53, 29: 0.53, 47: 0.53}, 0.52) == "thua"


def test_select_learner_view_prefers_qd3_within_tolerance():
    order = ["L0", "L1", "L2", "L3"]
    tied = {"L0": 0.540, "L1": 0.512, "L2": 0.5105, "L3": 0.509}   # L2, L3 chênh < 0,002
    assert select_learner_view(tied, order) == "L2"
    clear = {"L0": 0.540, "L1": 0.530, "L2": 0.515, "L3": 0.509}
    assert select_learner_view(clear, order) == "L3"
    with pytest.raises(ValueError):
        select_learner_view({"L9": 0.5}, order)


def _calibrated(n=60000, prior=(0.01, 0.27, 0.72), seed=0):
    """Ba lớp với xác suất hậu nghiệm đúng (mô hình hiệu chuẩn hoàn hảo)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=prior)
    mu = np.array([2.2, 1.0, 0.0])
    post = np.exp(-0.5 * (rng.normal(mu[y], 1.0)[:, None] - mu) ** 2) * np.array(prior)
    return y, post / post.sum(1, keepdims=True), np.array(prior)


def test_shrinking_toward_prior_keeps_auc_but_cuts_argmax_macro_f1():
    """Minh họa cho R5 (và một cơ chế khả dĩ của R4): nén xác suất về prior giữ nguyên
    thứ hạng (AUC) nhưng argmax ít chọn lớp G, nên Macro-F1 giảm mạnh; log loss tăng ít."""
    y, p, pi = _calibrated()
    z = np.log(p) / 1.5 + (1 - 1 / 1.5) * np.log(pi)
    q = np.exp(z - z.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    auc = lambda r: roc_auc_score(y, r, multi_class="ovr", average="macro")
    f1 = lambda r: f1_score(y, r.argmax(1), average="macro")
    rec_g = lambda r: recall_score(y, r.argmax(1), labels=[1], average="macro")
    assert abs(auc(q) - auc(p)) < 1e-3
    assert f1(p) - f1(q) > 0.04 and rec_g(q) < 0.6 * rec_g(p)
    assert 0 < log_loss(y, q) - log_loss(y, p) < 0.02
    cal = calibration_in_the_large(q, y)
    assert abs(cal["G"]["mean_proba"] - cal["G"]["observed"]) < 0.02
    assert cal["G"]["argmax_share"] < 0.5 * cal["G"]["observed"]


def test_subgroup_log_loss_and_code_gain_share():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    code = rng.integers(1, 200, 6000).astype(float)        # mã định danh nhóm
    effect = rng.normal(0, 1.5, 200)[code.astype(int)]      # nhãn phụ thuộc nhóm
    x = rng.normal(size=6000)
    y = np.digitize(0.3 * x + effect, [-1.0, 1.0])          # 0, 1, 2
    X = np.c_[x, code]
    m = xgb.XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.1, random_state=0)
    m.fit(X, y)
    names = ["behaviour", "teacher_set_code"]
    share = code_gain_share(m.get_booster(), names, ["teacher_set_code"])
    assert share > 0.5
    groups = np.where(code < 100, "seen", "unseen")
    out = subgroup_log_loss(m.predict_proba(X), y, groups)
    assert set(out) == {"seen", "unseen"} and out["seen"]["n"] + out["unseen"]["n"] == 6000


def test_course_contributions_sum_to_total_difference():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 3000)
    small = rng.choice(["c1", "c2", "c3", "c4"], 3000)
    course = np.where(np.arange(3000) < 1800, "big", small)
    pa, pb = rng.dirichlet([2, 2, 2], 3000), rng.dirichlet([2, 2, 2], 3000)
    big = course == "big"
    pb[big] = 0.98 * np.eye(3)[y[big]] + 0.01                     # B tốt hẳn ở khóa lớn
    la, lb = row_log_loss(pa, y), row_log_loss(pb, y)
    out = course_contributions(la, lb, course, top=5)
    assert out["total"] == pytest.approx(la.mean() - lb.mean())
    assert sum(out["top"].values()) == pytest.approx(out["total"])
    assert next(iter(out["top"])) == "big" and out["top_share"] == pytest.approx(1.0)
