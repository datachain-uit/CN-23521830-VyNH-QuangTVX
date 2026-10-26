"""Validated Review 25 October decisions and exact weighted family-bootstrap CI.

The unmodified teacher appendix is data/cq_patch_1025.py. These wrappers add
input guards and a rank-weighted CI equivalent to resampling whole families,
without sorting the same probabilities anew for each of 2,000 replicates.
No function approves a protocol, reads test, or trains a model.
"""
from __future__ import annotations

import numpy as np

from ..data.cq_patch_1025 import e_class_report, verdict_a8
from .cq_decisions_1023 import headroom_proposal, SELECTOR_SEEDS


DECISION_ENDPOINT = "mean_log_loss_A8_P1_P4"


def probabilities(value, y=None):
    p = np.asarray(value, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ValueError("Need finite normalized E/G/A probabilities")
    if y is not None:
        yy = np.asarray(y)
        if yy.shape != (len(p),) or not np.isin(yy, [0, 1, 2]).all():
            raise ValueError("Unaligned or invalid E/G/A labels")
    return p / p.sum(1, keepdims=True)


def verdict_proposal(a, b, delta, raw_a, raw_b, *, historical=False, phase_a=None, phase_b=None):
    """Require all registered dev selector seeds, record both verdicts and no approval."""
    from ..data.cq_patch_1026 import PANEL_SEEDS, verdict_panel
    required = {11, 29, 47} if historical else set(PANEL_SEEDS)
    if set(a) != required or set(raw_a) != set(a):
        raise ValueError(f"Decision requires registered selector seeds {sorted(required)}")
    for values in (a, b, raw_a, raw_b):
        arr = np.asarray(list(values.values()) if isinstance(values, dict) else [values], float)
        if not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError("Invalid loss values")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Margin must be finite and positive")
    out = (verdict_a8(a, b, delta, raw_a, raw_b) if historical else
           verdict_panel(a, b, delta, raw_a, raw_b, phase_a, phase_b))
    out["rule_revision"] = "Review-2026-10-25" if historical else "Review-2026-10-26"
    out.update(approval_status="pending_teacher", test_data_read=False)
    return out


def choose_fixed_budget_baseline(records, *, historical=False):
    """K1: lowest mean decision loss, exclude C03 and native-size runs explicitly."""
    eligible = []
    for record in records:
        if record["method_id"] == "c03_phase_class_random" or record.get("role") == "C03":
            continue
        if record.get("budget_mode") != "fixed":
            continue
        scores = record["ul_decision_by_seed"]
        from ..data.cq_patch_1026 import PANEL_SEEDS
        required = {11, 29, 47} if historical else set(PANEL_SEEDS)
        if set(scores) != required:
            raise ValueError(f"Baseline selection requires seeds {sorted(required)}")
        values = np.asarray(list(scores.values()), float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid baseline scores")
        eligible.append((float(values.mean()), record["method_id"]))
    if not eligible:
        raise ValueError("No eligible fixed-budget baseline")
    return {"method_id": min(eligible)[1], "decision_quantity": "UL_A8_or_flagged_raw_fallback",
            "approval_status": "pending_teacher", "excluded_C03_and_native": True}


def selection_phase_priors(y_full, indices, method_weights, phases, class_weight_policy):
    """QD15 effective selected priors conditional on phase, not the pooled prior.

The conditional source priors measure H3d dp; A8 itself retains the teacher's
global fitted-model source prior. An absent selected phase stays undefined.
"""
    from ..training.weights import balanced_class_weights, effective_sample_weights
    y = np.asarray(y_full)[indices]
    phase = np.asarray(phases)[indices]
    cw = balanced_class_weights(y_full) if class_weight_policy == "balanced" else None
    weights = effective_sample_weights(y, method_weights, cw)
    out = {}
    for number, name in enumerate(("P1", "P2", "P3", "P4")):
        mask = phase == number
        mass = weights[mask].sum()
        out[name] = {lab: float(weights[mask & (y == k)].sum()/mass) for k, lab in enumerate("EGA")} if mass > 0 else None
    return out


def headroom_a8_proposal(full, corrected_c02, raw_c02, prior_ul):
    """Keep QD26 grid/seed/margin guards; use A8 C02, report raw in parallel."""
    corrected = headroom_proposal(full, corrected_c02, prior_ul)
    raw = headroom_proposal(full, raw_c02, prior_ul)
    for ratio, row in corrected["rows"].items():
        row["raw_comparison"] = raw["rows"][ratio]
    corrected.update(primary_endpoint=DECISION_ENDPOINT,
                     decision_quantity="UL_A8", raw_rho_star=raw["rho_star"],
                     prior_dependent=corrected["rho_star"] != raw["rho_star"],
                     rule_revision="Review-2026-10-25_QD30")
    return corrected


def e_family_ci(proba, y, families, *, n_bootstrap=2000, seed=20261025):
    """Exact whole-family bootstrap for pooled E NLL/AUC/AP, including score ties.

With concatenated P1-P4, each sampled family carries ALL its phases. Classifier
probabilities stay fixed; rows are weighted by the family's resample count.
NLL clips as in the teacher appendix, no-E/only-E samples are recorded invalid.
"""
    p = probabilities(proba, y)
    y = np.asarray(y)
    fam = np.asarray(families).astype(str)
    if fam.shape != y.shape or n_bootstrap < 2:
        raise ValueError("Need aligned families and at least two bootstrap draws")
    keys, inverse = np.unique(fam, return_inverse=True)
    is_e = y == 0
    score = p[:, 0]
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    pos = is_e[order]
    group = inverse[order]
    e_counts = np.bincount(inverse, weights=is_e, minlength=len(keys))
    losses = np.where(is_e, -np.log(np.clip(score, 1e-15, 1)), 0)
    sums = np.bincount(inverse, weights=losses, minlength=len(keys))
    draws = {m: [] for m in ("nll_E", "auc_E", "ap_E")}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        mult = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
        n_e = mult @ e_counts
        if n_e > 0:
            draws["nll_E"].append(float((mult @ sums) / n_e))
        weights = mult[group]
        tp = np.add.reduceat(weights * pos, starts).astype(float)
        fp = np.add.reduceat(weights * (~pos), starts).astype(float)
        n_pos, n_neg = tp.sum(), fp.sum()
        if n_pos > 0 and n_neg > 0:
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            precision = np.divide(cum_tp, cum_tp + cum_fp,
                                  out=np.zeros_like(cum_tp), where=(cum_tp + cum_fp) > 0)
            draws["ap_E"].append(float(np.sum((tp / n_pos) * precision)))
            draws["auc_E"].append(float(np.sum(fp * (cum_tp - .5 * tp)) / (n_pos * n_neg)))
    point = e_class_report(p, y, fam)
    return {m: {"point": point[m],
                "ci95": np.quantile(v, [.025, .975]).tolist() if v else None,
                "n_valid": len(v), "n_bootstrap": n_bootstrap,
                "bootstrap_unit": "course_family_id", "phase_resampling": "joint",
                "implementation": "exact_weighted_family_resample"}
            for m, v in draws.items()}


def e_phase_report(proba_by_phase, labels_by_phase, families_by_phase, *, n_bootstrap=2000):
    phases = ("P1", "P2", "P3", "P4")
    if any(set(m) != set(phases) for m in (proba_by_phase, labels_by_phase, families_by_phase)):
        raise ValueError("E report requires complete P1-P4")
    p = {ph: probabilities(proba_by_phase[ph], labels_by_phase[ph]) for ph in phases}
    per = {ph: e_class_report(p[ph], labels_by_phase[ph], families_by_phase[ph]) for ph in phases}
    pooled_p = np.concatenate([p[ph] for ph in phases])
    pooled_y = np.concatenate([labels_by_phase[ph] for ph in phases])
    pooled_fam = np.concatenate([families_by_phase[ph] for ph in phases])
    return {"by_phase": per, "pooled_four_phases": e_class_report(pooled_p, pooled_y, pooled_fam),
            "pooled_ci95": e_family_ci(pooled_p, pooled_y, pooled_fam, n_bootstrap=n_bootstrap),
            "pooled_rows_are_not_independent_enrollments": True}
