"""Validated log-loss decisions and paired family bootstrap for Review 23/10.

The teacher's Appendix A is preserved separately in data/cq_patch_1023.py.
These wrappers enforce complete registered seeds/grids before making a proposal;
no function here grants approval, opens test or chooses a new feature view.
"""
from __future__ import annotations

import numpy as np

from ..data.cq_patch_1023 import skill_margin, headroom_rho_ll, row_log_loss, PHASES


RATIOS = (0.005, 0.01, 0.02, 0.05)
SELECTOR_SEEDS = (11, 29, 47, 71, 101)
MODEL_SEEDS = (42, 43, 44)


def _losses(values) -> np.ndarray:
    array = np.asarray(list(values), float)
    if not array.size or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Log losses must be finite and nonnegative (not restricted to [0,1])")
    return array


def headroom_proposal(full_by_model_seed: dict, c02_by_ratio: dict, prior_ul: float) -> dict:
    if set(full_by_model_seed) != set(MODEL_SEEDS):
        raise ValueError("FullTrain-B requires model seeds 42/43/44")
    if set(c02_by_ratio) != set(RATIOS):
        raise ValueError("Headroom requires budget grid 0.005/0.01/0.02/0.05")
    for ratio, values in c02_by_ratio.items():
        if set(values) != set(SELECTOR_SEEDS):
            raise ValueError(f"C02 at {ratio} requires all five selector seeds")
        _losses(values.values())
    full = _losses(full_by_model_seed.values())
    _losses([prior_ul])
    mean = float(full.mean())
    margin = skill_margin(prior_ul, mean)
    result = headroom_rho_ll(mean, c02_by_ratio, gap=2 * margin, grid=RATIOS)
    result.update({
        "primary_endpoint": "mean_log_loss_P1_P4", "direction": "lower_is_better",
        "delta_ll_proposed": margin, "prior_ul": float(prior_ul),
        "fulltrain_b_mean": mean, "fulltrain_b_sd": float(full.std(ddof=1)),
        "fulltrain_b_model_seeds": list(MODEL_SEEDS), "selector_seeds": list(SELECTOR_SEEDS),
        "rule_revision": "Review-2026-10-23_QD25_QD26", "approval_status": "pending_teacher",
        "test_data_read": False,
    })
    return result


def select_by_loss_and_time(scores: dict, seconds: dict, tol: float = 0.002) -> str:
    """Stage-2/LR rule; L0/L1 preference instead uses teacher select_learner_view."""
    if set(scores) != set(seconds) or not scores or tol < 0:
        raise ValueError("Need matching nonempty scores/times and nonnegative tolerance")
    _losses(scores.values())
    _losses(seconds.values())
    minimum = min(scores.values())
    eligible = [key for key in scores if scores[key] <= minimum + tol]
    return min(eligible, key=lambda key: (seconds[key], key))


def paired_family_logloss_ci(proba_a: dict, proba_b: dict, labels: dict, families: dict,
                             *, n_bootstrap: int = 2000, seed: int = 20261023) -> dict:
    """Resample families ONCE per replicate jointly across all four phases.

    Within each phase the estimand remains enrollment-weighted mean log loss;
    it is not changed into a mean of family means. Callers must align row IDs.
    """
    for mapping in (proba_a, proba_b, labels, families):
        if set(mapping) != set(PHASES):
            raise ValueError("Paired bootstrap requires exactly P1–P4")
    if n_bootstrap < 2:
        raise ValueError("Need at least two bootstrap draws")
    keys = np.unique(np.asarray(families["P1"]).astype(str))
    if not len(keys):
        raise ValueError("No family groups")
    sums, counts, point = [], [], []
    for phase in PHASES:
        y = np.asarray(labels[phase])
        group = np.asarray(families[phase]).astype(str)
        if set(group) != set(keys) or len(group) != len(y):
            raise ValueError("Family cohort differs across phases")
        for prob in (proba_a[phase], proba_b[phase]):
            p = np.asarray(prob)
            if (p.shape != (len(y), 3) or not np.isfinite(p).all()
                    or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
                raise ValueError("Invalid aligned E/G/A probabilities")
        delta = row_log_loss(proba_a[phase], y) - row_log_loss(proba_b[phase], y)
        positions = np.searchsorted(keys, group)
        sums.append(np.bincount(positions, weights=delta, minlength=len(keys)))
        counts.append(np.bincount(positions, minlength=len(keys)))
        point.append(float(delta.mean()))
    sums, counts = np.asarray(sums).T, np.asarray(counts).T
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 100):
        k = min(100, n_bootstrap - start)
        weights = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=k)
        draws[start:start+k] = ((weights @ sums) / (weights @ counts)).mean(axis=1)
    return {"mean_diff_a_minus_b": float(np.mean(point)),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "n_bootstrap": n_bootstrap, "seed": seed, "n_families": len(keys),
            "bootstrap_unit": "course_family_id", "phase_resampling": "joint",
            "estimand": "enrollment_weighted_LL_equal_mean_of_four_phases"}
