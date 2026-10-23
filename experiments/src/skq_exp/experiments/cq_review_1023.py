"""Runtime guards and dev-only data routing for Review 2026-10-23.

L0 stops on dev P4; L1 stops on the concatenation of four dev phases.
S is a diagnostic fit on train P4, never an alternative cohort/split.
Explicit feature-view checks prevent V4 ready arrays being labelled CD3.
"""
from __future__ import annotations

import numpy as np
from ..data.cq_patch_1023 import calibration_in_the_large
from ..training.metrics import compute_overall_metrics, compute_per_label_metrics


def guard_feature_view(config, data: dict) -> None:
    expected = getattr(config, "feature_view", None)
    if expected is None:
        return
    contract = data.get("feature_contract") or {}
    actual = contract.get("feature_view", "v4")
    if actual != expected:
        raise ValueError(f"feature_view mismatch: requested={expected}, prepared={actual}")
    if not config.test_locked:
        raise ValueError("Review 23/10 requires test_locked=true")
    if any(key.startswith(("X_test", "y_test")) for key in data):
        raise ValueError("Review 23/10 runtime must be dev-only, not contain test arrays")


def diagnostic_training_data(config, data: dict) -> dict:
    """Return a non-mutating P4 slice only for explicitly registered diagnostic S."""
    phases = getattr(config, "training_phases", ())
    if not phases:
        return data
    if tuple(phases) != ("P4",):
        raise ValueError("Only diagnostic S train P4 is supported")
    phase = np.asarray(data["phase_train"])
    mask = phase == 3  # Prepared CQ adapter maps P1..P4 to 0..3.
    if not mask.any():
        raise ValueError("Diagnostic S has no train P4 rows")
    out = dict(data)
    for key, value in data.items():
        if key.endswith("_train") and "fulltrain_a" not in key and isinstance(value, np.ndarray):
            if len(value) != len(mask):
                raise ValueError(f"Train metadata alignment mismatch: {key}")
            out[key] = np.asarray(value)[mask]
    return out


def early_stopping_data(config, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate requested dev phases into ONE eval_set; never use last-only XGB lists."""
    policy = config.early_stopping or {}
    phases = policy.get("phases")
    if phases is None:
        return np.asarray(data["X_dev"]), np.asarray(data["y_dev"])
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("Early-stopping phases must be nonempty and unique")
    xs, ys = [], []
    for phase in phases:
        if phase not in {"P1", "P2", "P3", "P4"}:
            raise ValueError("Early stopping only accepts dev phases P1–P4")
        key = f"dev_phase{phase[-1]}"
        X, y = np.asarray(data[f"X_{key}"]), np.asarray(data[f"y_{key}"])
        if len(X) != len(y) or not len(y):
            raise ValueError(f"Early-stopping {phase}: missing/unaligned rows")
        xs.append(X)
        ys.append(y)
    if len({len(y) for y in ys}) > 1:
        raise ValueError("L1 requires equal-sized dev phases for equal phase weighting")
    return (xs[0], ys[0]) if len(xs) == 1 else (np.concatenate(xs), np.concatenate(ys))


def xgb_round_diagnostics(model, X, y, requested) -> dict:
    """Iteration ranges are boosting rounds (three trees/round for CQ multiclass).

    Never request beyond fitted rounds or relabel a shorter model as 400 rounds.
    Diagnostic prediction cost is separate from the ordinary prediction timing.
    """
    import time
    started = time.perf_counter()
    available = int(model.get_booster().num_boosted_rounds())
    try:
        best = int(model.best_iteration) + 1
    except (AttributeError, TypeError):
        best = available
    targets = {str(int(r)): int(r) for r in requested}
    targets["best"] = best
    points = {}
    for name, rounds in targets.items():
        if rounds < 1 or rounds > available:
            points[name] = {"status": "not_available", "boosting_rounds_requested": rounds,
                            "boosting_rounds_fitted": available}
            continue
        p = model.predict_proba(X, iteration_range=(0, rounds))
        pred = np.argmax(p, axis=1)
        overall, warnings = compute_overall_metrics(y, pred, p, labels=np.arange(3))
        points[name] = {"status": "success", "boosting_rounds_used": rounds,
                        "trees_used": rounds * 3, "overall": overall,
                        "per_label": compute_per_label_metrics(y, pred, p, np.arange(3)),
                        "calibration_in_the_large": calibration_in_the_large(p, y),
                        "warnings": warnings}
    return {"points": points, "diagnostic_predict_seconds": time.perf_counter()-started,
            "iteration_range_unit": "boosting_rounds", "not_used_for_hyperparameter_selection": True}
