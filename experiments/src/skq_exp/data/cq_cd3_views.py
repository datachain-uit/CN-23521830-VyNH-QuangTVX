"""Build train-only CD3 input views on an immutable V4 enrollment spine.

Consumes raw features already built by the data team, not raw events or labels
from CD3. QA23 uses only Pool B train; one V1-Median transform is shared by
train, all four dev phases, and FullTrain-A. This does not approve QA columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import stable_hash
from .cq_tempo_features import (
    CANDIDATES, PHASE_FRACTION, CQRawEarlyViews, add_hour_cyclical,
    raw_early_columns, qa23_audit, held_out_columns, STATIC_FEATURES,
)


PHASES = ("P1", "P2", "P3", "P4")
META = {"row_id", "enrollment_id", "user_id", "course_id", "course_family_id",
        "phase", "split", "y_alpha_1", "phase_fraction"}
MASKS = {f"{family}_observed_mask" for family in ("video", "problem", "comment")}


def _check(frame: pd.DataFrame, name: str, split: str, phases: set[str]) -> None:
    missing = (set(CANDIDATES) | META | MASKS) - set(frame)
    if missing:
        raise ValueError(f"{name}: raw CD3 missing columns {sorted(missing)}")
    if frame.empty or frame[list(META - {"phase_fraction"})].isna().any().any():
        raise ValueError(f"{name}: missing row/label/group metadata")
    if frame.row_id.duplicated().any() or frame.duplicated(["enrollment_id", "phase"]).any():
        raise ValueError(f"{name}: duplicate snapshot")
    if set(frame.split) != {split} or set(frame.phase) != phases:
        raise ValueError(f"{name}: split/phase mismatch")
    expected = frame.phase.map(PHASE_FRACTION).to_numpy(float)
    if not np.allclose(frame.phase_fraction.to_numpy(float), expected):
        raise ValueError(f"{name}: phase_fraction differs from V4")
    if not set(frame.y_alpha_1) <= {"E", "G", "A"}:
        raise ValueError(f"{name}: label outside E/G/A")
    if not frame[list(MASKS)].isin([0, 1]).all().all():
        raise ValueError(f"{name}: observation masks must be 0/1 audit columns")


def build_cd3_views(train: pd.DataFrame, dev: dict[str, pd.DataFrame], full_a: pd.DataFrame,
                    *, feature_variant: str = "main"):
    """Return ready frames, QA audit, feature contract and serializable fit parameters."""
    if feature_variant not in {"main", "qa23_all"}:
        raise ValueError("Unknown CD3 feature_variant")
    _check(train, "train", "train", set(PHASES))
    if set(dev) != set(PHASES):
        raise ValueError("CD3 needs exactly four dev phases")
    if "q_alpha_1" not in train:
        raise ValueError("Train q_alpha_1 is required for QA23, never as a feature")
    q = pd.to_numeric(train.q_alpha_1, errors="coerce").to_numpy(float)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Train q_alpha_1 must be finite in [0,1]")
    y = np.select([q < 0.2, q < 0.4], ["E", "G"], default="A")
    if not np.array_equal(y, train.y_alpha_1.to_numpy()):
        raise ValueError("Train y != tau(q)")
    groups = train.groupby("enrollment_id", sort=False)
    if not groups.size().eq(4).all():
        raise ValueError("Train enrollment must have exactly four snapshots")
    stable = ["user_id", "course_id", "course_family_id", "y_alpha_1", "q_alpha_1"]
    if groups[stable].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Train enrollment metadata/target changes between phases")
    if groups[list(STATIC_FEATURES)].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Static CD3 features change across snapshots of one enrollment")
    truth = train.drop_duplicates("enrollment_id").set_index("enrollment_id")
    _check(full_a, "FullTrain-A", "train", {"FULL_A"})
    if set(full_a.enrollment_id) != set(truth.index):
        raise ValueError("FullTrain-A does not match train cohort")
    for col in ["user_id", "course_id", "course_family_id", "y_alpha_1"]:
        if not np.array_equal(full_a[col].to_numpy(), truth.loc[full_a.enrollment_id, col].to_numpy()):
            raise ValueError(f"FullTrain-A {col} differs from V4 train")
    base_dev = None
    for phase in PHASES:
        frame = dev[phase]
        _check(frame, phase, "dev", {phase})
        if set(frame.course_family_id) & set(train.course_family_id):
            raise ValueError("Train/dev family leakage")
        if set(frame.enrollment_id) & set(train.enrollment_id):
            raise ValueError("Train/dev enrollment leakage")
        current = frame.set_index("enrollment_id")[stable[:-1]].sort_index()
        if base_dev is None:
            base_dev = current
        elif not current.equals(base_dev):
            raise ValueError("Dev cohort/metadata/target differs across phases")
    numeric = raw_early_columns(add_hour_cyclical(train))["selector"]
    if len(numeric) != 74:
        raise ValueError(f"Expected 74 numeric before QA23, got {len(numeric)}")
    audit = qa23_audit(add_hour_cyclical(train), "q_alpha_1", numeric)
    flagged = held_out_columns(audit)
    drop = [] if feature_variant == "qa23_all" else flagged
    views = CQRawEarlyViews().fit(train, drop=drop)
    frames = {"train": train, **{f"dev_phase{p[-1]}": dev[p] for p in PHASES}, "fulltrain_a": full_a}
    ready = {}
    for name, frame in frames.items():
        matrix = views.transform(frame)["learner"]
        if not np.isfinite(matrix.to_numpy(float)).all():
            raise ValueError(f"{name}: nonfinite ready features")
        # Learner columns are overwritten by V1 values under the original names.
        # Metadata/non-learner columns remain; D06/QA23 must read raw_features,
        # never treat these ready columns as untransformed event aggregates (Y30).
        ready[name] = frame.copy()
        ready[name][matrix.columns] = matrix
    prep = {
        "fit_scope": "TRAIN_POOLED_P1_P4", "static_scope": "once_per_enrollment",
        "center": views.v1.center, "scale": views.v1.scale,
        "log_columns": views.v1.log_columns, "category_maps": views.maps,
        "held_out_columns": drop,
        "train_row_hash": stable_hash(train.row_id.tolist()),
    }
    contract = {
        "feature_view": "cd3", "selector_columns": views.cols["selector"],
        "feature_variant": feature_variant, "qa23_flagged_columns": flagged,
        "ready_learner_representation": "V1_Median_under_original_column_names",
        "learner_columns": views.cols["learner"], "held_out_columns": drop,
        "gender_missing_code": views.maps.get("gender", {}).get("<MISSING>"),
        "preprocessing_scope": "TRAIN_POOLED_P1_P4_V1_MEDIAN",
        "qa23_scope": "train_only_per_phase", "qa23_approved_by_teacher": False,
        "preprocessing_hash": stable_hash(prep),
    }
    contract["contract_hash"] = stable_hash(contract)
    return ready, audit, contract, prep
