"""Regression tests for dev-only D13 routing, LR encoding and review decisions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skq_exp.config import ExperimentConfig
from skq_exp.data.cq_cd3_views import build_cd3_views
from skq_exp.data.cq_lr_views import lr_onehot_views
from skq_exp.data.cq_tempo_features import CANDIDATES, PHASE_FRACTION, STATIC_FEATURES
from skq_exp.experiments.cq_review_1023 import (
    early_stopping_data, guard_feature_view, diagnostic_training_data,
)
from skq_exp.experiments.cq_decisions_1023 import (
    headroom_proposal, paired_family_logloss_ci, select_by_loss_and_time,
)
from skq_exp.experiments.provenance import protocol_hash
from skq_exp.experiments.runner import _evaluation_splits
from scripts.materialize_cq_review2310_configs import configs_for_stage


def _raw(n=30, split="train", seed=1):
    rng = np.random.default_rng(seed)
    q = np.linspace(0.01, 0.9, n)
    base = pd.DataFrame(rng.random((n, len(CANDIDATES))), columns=CANDIDATES)
    base["gender"] = np.where(np.arange(n) % 3, 1, np.nan)
    base["enrollment_id"] = [f"{split}-e{i}" for i in range(n)]
    base["user_id"] = [f"{split}-u{i}" for i in range(n)]
    base["course_id"] = f"{split}-course"
    base["course_family_id"] = f"{split}-family"
    base["split"] = split
    base["q_alpha_1"] = q
    base["y_alpha_1"] = np.select([q < .2, q < .4], ["E", "G"], default="A")
    for family in ("video", "problem", "comment"):
        base[f"{family}_observed_mask"] = 1
    phases = []
    for phase in ("P1", "P2", "P3", "P4"):
        frame = base.copy()
        frame["phase"] = phase
        frame["phase_fraction"] = PHASE_FRACTION[phase]
        frame["row_id"] = frame.enrollment_id + "-" + phase
        frame["enroll_time"] = pd.Timestamp("2020-01-01")
        frame["cutoff_time"] = pd.Timestamp("2020-01-01") + pd.Timedelta(days=10*PHASE_FRACTION[phase])
        phases.append(frame)
    return pd.concat(phases, ignore_index=True)


def _frames():
    train, pooled_dev = _raw(), _raw(split="dev", seed=2)
    dev = {p: pooled_dev[pooled_dev.phase == p].copy() for p in ("P1", "P2", "P3", "P4")}
    full = train[train.phase == "P4"].copy()
    full["phase"], full["phase_fraction"] = "FULL_A", 1.0
    full["row_id"] = full.enrollment_id + "-FULL_A"
    full["cutoff_time"] = pd.Timestamp("2020-01-11")
    return train, dev, full


def test_cd3_fit_does_not_use_dev_and_retains_metadata():
    train, dev, full = _frames()
    before = train.copy(deep=True)
    ready, audit, contract, prep = build_cd3_views(train, dev, full)
    altered = {p: f.assign(**{CANDIDATES[0]: 9999999}) for p, f in dev.items()}
    _, _, contract2, prep2 = build_cd3_views(train, altered, full)
    assert prep == prep2 and contract == contract2
    assert set(audit.phase) == {"P1", "P2", "P3", "P4"}
    assert contract["feature_view"] == "cd3"
    assert contract["learner_columns"] == contract["selector_columns"] + ["gender", "phase_fraction"]
    assert not (set(contract["held_out_columns"]) & set(contract["learner_columns"]))
    for frame in ready.values():
        assert np.isfinite(frame[contract["learner_columns"]].to_numpy()).all()
    assert ready["train"].row_id.tolist() == train.row_id.tolist()
    pd.testing.assert_frame_equal(before, train)


@pytest.mark.parametrize("mutation", ["family", "target", "missing", "cohort", "duplicate", "mask", "static"])
def test_cd3_rejects_corrupt_spine(mutation):
    train, dev, full = _frames()
    if mutation == "family":
        dev["P1"]["course_family_id"] = train.course_family_id.iloc[0]
    elif mutation == "target":
        train.loc[0, "y_alpha_1"] = "A"
    elif mutation == "missing":
        train = train.drop(columns=[CANDIDATES[0]])
    elif mutation == "cohort":
        full = full.iloc[:-1]
    elif mutation == "mask":
        train = train.drop(columns=["video_observed_mask"])
    elif mutation == "static":
        train.loc[0, "teacher_count"] += 1
    else:
        train.loc[1, "row_id"] = train.loc[0, "row_id"]
    with pytest.raises(ValueError):
        build_cd3_views(train, dev, full)


def test_early_stop_L1_concatenates_one_dev_set_without_test():
    data = {f"{stem}_dev_phase{i}": np.full((2, 3) if stem == "X" else 2, i)
            for i in range(1, 5) for stem in ("X", "y")}
    config = SimpleNamespace(early_stopping={"phases": ["P1", "P2", "P3", "P4"]})
    X, y = early_stopping_data(config, data)
    assert X.shape == (8, 3) and y.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    config.early_stopping["phases"] = ["test"]
    with pytest.raises(ValueError, match="only accepts dev"):
        early_stopping_data(config, data)


def test_feature_guard_cannot_relabel_V4_as_CD3_or_accept_test():
    config = SimpleNamespace(feature_view="cd3", test_locked=True)
    with pytest.raises(ValueError, match="mismatch"):
        guard_feature_view(config, {"feature_contract": {}})
    with pytest.raises(ValueError, match="test arrays"):
        guard_feature_view(config, {"feature_contract": {"feature_view": "cd3"}, "X_test": np.ones(1)})


def test_S_is_nonmutating_P4_slice_and_only_evaluates_P4():
    config = SimpleNamespace(training_phases=("P4",), stage_id="s1_screen")
    data = {"phase_train": np.arange(4), "X_train": np.arange(8).reshape(4, 2),
            "y_train": np.arange(4), "row_ids_train": np.arange(4)}
    data.update({f"{stem}_dev_phase{i}": np.ones(2) for i in range(1, 5) for stem in ("X", "y")})
    selected = diagnostic_training_data(config, data)
    assert len(data["X_train"]) == 4 and selected["row_ids_train"].tolist() == [3]
    assert _evaluation_splits(config, "course_quality_v4", selected) == ("dev_phase4",)


def test_LR_gender_encoder_uses_all_train_and_keeps_NULL_distinct():
    gender = np.array([1]*20 + [2]*3 + [3]*2)
    train = np.column_stack([np.arange(25), gender, np.ones(25)])
    data = {"X_train": train, "X_dev_phase1": np.array([[0, 2, 1], [0, 99, 1], [0, 3, 1]]),
            "X_selector_train": train[:, :1],
            "feature_contract": {"feature_view": "cd3", "gender_missing_code": 3,
                                 "learner_columns": ["numeric", "gender", "phase_fraction"]}}
    out, metadata = lr_onehot_views(data)
    assert metadata["kept_gender_codes"] == [1]
    assert metadata["fit_scope"] == "all_pool_b_train"
    assert out["X_dev_phase1"][:, -3:].tolist() == [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert out["X_selector_train"] is data["X_selector_train"]


def test_headroom_requires_complete_seeds_and_allows_LL_above_one():
    full = {42: 1.1, 43: 1.1, 44: 1.1}
    c02 = {r: {s: 1.3 for s in (11, 29, 47, 71, 101)} for r in (.005, .01, .02, .05)}
    result = headroom_proposal(full, c02, 1.5)
    assert result["rho_star"] == .05 and result["delta_ll_proposed"] == .02
    assert result["approval_status"] == "pending_teacher"
    c02[.005].pop(101)
    with pytest.raises(ValueError, match="all five"):
        headroom_proposal(full, c02, 1.5)


def test_loss_tie_selects_fastest_not_last_or_first_config():
    assert select_by_loss_and_time({"a": .501, "b": .5}, {"a": 10., "b": 20.}) == "a"


def test_family_bootstrap_preserves_phase_pairing_and_enrollment_estimand():
    y = np.tile(np.arange(3), 4)
    pa = np.tile([.2, .3, .5], (12, 1))
    pb = np.tile([.3, .3, .4], (12, 1))
    families = np.array(["large"]*9 + ["small"]*3)
    mappings = [{p: val for p in ("P1", "P2", "P3", "P4")} for val in (pa, pb, y, families)]
    out = paired_family_logloss_ci(*mappings, n_bootstrap=100)
    expected = (-np.log(pa[np.arange(12), y]) + np.log(pb[np.arange(12), y])).mean()
    assert out["mean_diff_a_minus_b"] == pytest.approx(expected)
    assert out["phase_resampling"] == "joint"
    same = paired_family_logloss_ci(mappings[0], mappings[0], mappings[2], mappings[3], n_bootstrap=100)
    assert same["ci95"] == [0., 0.]


def test_config_hash_changes_with_view_and_rejects_arbitrary_train_phase():
    base = ExperimentConfig.from_json(Path(__file__).parents[1] / "configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json")
    a = replace(base, feature_view="cd3")
    assert protocol_hash(a) != protocol_hash(base)
    with pytest.raises(ValueError, match="diagnostic S"):
        replace(a, training_phases=("P1",)).validate()


def test_all_fourteen_registered_configs_validate_and_preserve_old_configs():
    root = Path(__file__).parents[1]
    configs = list((root / "configs").glob("cq_v4_review2310_*_v1.json"))
    assert len(configs) == 14
    for path in configs:
        config = ExperimentConfig.from_json(path)
        assert config.test_locked and config.artifact_policy["save_predictions_parquet"]
        assert "review2310" in str(config.paths.artifact_root)
        assert config.model_seed == 42
    old = ExperimentConfig.from_json(root / "configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json")
    assert old.feature_view is None and old.early_stopping["patience"] == 20
    assert "feature_view" not in old.to_dict() and "label_shuffle_seed" not in old.to_dict()


def test_registered_materialization_has_six_plus_two_and_complete_followups():
    base = ExperimentConfig.from_json(Path(__file__).parents[1] / "configs/cq_v4_review2310_l1_v1.json")
    original = base.to_dict()
    grid = configs_for_stage(base, "grid")
    assert len(grid) == 6
    assert {(c.learner_params["xgb"]["max_depth"], c.learner_params["xgb"]["learning_rate"]) for c in grid.values()} == {
        (d, lr) for d in (4, 6, 8) for lr in (.05, .1)}
    assert len(configs_for_stage(next(iter(grid.values())), "min-child")) == 2
    following = configs_for_stage(base, "followups")
    assert len(following) == 11
    assert {c.label_shuffle_seed for c in following.values() if c.label_shuffle_seed is not None} == {20261022, 20261023, 20261024}
    assert {c.model_seed for n, c in following.items() if n.startswith("fullB")} == {42, 43, 44}
    c02 = [c for c in following.values() if c.method_ids == ("c02_stratified_random",)]
    assert len(c02) == 4 and all(c.selector_seeds == (11, 29, 47, 71, 101) for c in c02)
    assert base.to_dict() == original


def test_shuffle_seed_cannot_be_used_for_panel_or_non_diagnostic_evidence():
    base = ExperimentConfig.from_json(Path(__file__).parents[1] / "configs/cq_v4_review2310_l0_v1.json")
    with pytest.raises(ValueError, match="Y26"):
        replace(base, label_shuffle_seed=20261023).validate()
    with pytest.raises(ValueError, match="float32"):
        replace(base, artifact_policy={"save_predictions_parquet": True}).validate()
