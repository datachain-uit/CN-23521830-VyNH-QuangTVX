"""Review25 integration/edge checks on synthetic data only; never opens CQ test."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from skq_exp.config import ExperimentConfig
from skq_exp.data.cq_cd3_views import build_cd3_views
from skq_exp.data.cq_patch_1025 import family_bootstrap_ci
from skq_exp.experiments.cq_review_1023 import guard_feature_view
from skq_exp.experiments.cq_review_1025 import (
    e_family_ci, e_phase_report, headroom_a8_proposal, verdict_proposal,
    selection_phase_priors, choose_fixed_budget_baseline)
from skq_exp.experiments.provenance import protocol_hash
from scripts.materialize_cq_review2510_configs import followups
from test_cq_review2310_runtime import _frames


ROOT = Path(__file__).parents[1]


def test_dp_uses_conditional_selected_phase_prior_and_marks_absent_phase():
    y = np.array([0, 1, 2, 0, 1, 2])
    phase = np.array([0, 0, 0, 1, 1, 1])
    result = selection_phase_priors(y, np.array([0, 2, 4]), np.array([3., 1., 2.]), phase, "natural")
    assert result["P1"] == {"E": .75, "G": 0., "A": .25}
    assert result["P2"] == {"E": 0., "G": 1., "A": 0.}
    assert result["P3"] is None and result["P4"] is None


def test_K1_excludes_C03_and_native_even_when_their_score_is_lower():
    records = [{"method_id": name, "budget_mode": mode, "role": role,
                "ul_decision_by_seed": {s: loss for s in (11,29,47)}}
               for name, mode, role, loss in [("c03", "fixed", "C03", .1),
                    ("native", "native", "baseline", .2), ("random", "fixed", "baseline", .5)]]
    assert choose_fixed_budget_baseline(records)["method_id"] == "random"


@pytest.mark.parametrize("ties", [False, True])
def test_fast_CI_exactly_matches_teacher_explicit_resampling(ties):
    rng = np.random.default_rng(19)
    p = rng.dirichlet([1, 3, 5], 80)
    if ties:
        p = np.tile([[.1, .4, .5], [.4, .3, .3]], (40, 1))
    y = rng.integers(0, 3, 80)
    families = rng.integers(0, 12, 80)
    reference = family_bootstrap_ci(p, y, families, n_bootstrap=80, seed=23)
    actual = e_family_ci(p, y, families, n_bootstrap=80, seed=23)
    for metric in reference:
        assert actual[metric]["point"] == pytest.approx(reference[metric]["point"], abs=1e-12)
        assert actual[metric]["ci95"] == pytest.approx(reference[metric]["ci95"], abs=1e-12)
        assert actual[metric]["n_valid"] == reference[metric]["n_valid"]


def test_no_E_returns_undefined_CI_not_zero_score():
    report = e_family_ci(np.tile([.1, .4, .5], (8, 1)), np.repeat(2, 8), np.arange(8), n_bootstrap=10)
    assert all(v["ci95"] is None and v["n_valid"] == 0 for v in report.values())


def test_pooled_E_family_count_is_not_four_times_phase_count():
    p = np.tile([.6, .2, .2], (8, 1))
    maps = [{ph: value for ph in ("P1", "P2", "P3", "P4")}
            for value in (p, np.array([0, 0, 1, 1, 2, 2, 2, 2]), np.array(["f1"]*4 + ["f2"]*4))]
    result = e_phase_report(*maps, n_bootstrap=10)
    assert result["pooled_four_phases"]["n_families_with_E"] == 1
    assert result["pooled_four_phases"]["n_E"] == 8
    assert result["pooled_ci95"]["nll_E"]["phase_resampling"] == "joint"


def test_A8_headroom_selects_corrected_not_raw_and_keeps_grid_guard():
    full = {42: .5, 43: .5, 44: .5}
    raw = {r: {s: .65 for s in (11,29,47,71,101)} for r in (.005,.01,.02,.05)}
    a8 = {r: {s: .501 for s in (11,29,47,71,101)} for r in raw}
    result = headroom_a8_proposal(full, a8, raw, .7)
    assert result["rho_star"] == .005 and result["raw_rho_star"] == .05
    assert result["prior_dependent"] and result["decision_quantity"] == "UL_A8"
    assert result["rows"][.05]["raw_comparison"]["mean"] == pytest.approx(.65)
    a8[.005].pop(101)
    with pytest.raises(ValueError):
        headroom_a8_proposal(full, a8, raw, .7)


def test_verdict_requires_three_seeds_and_marks_prior_dependence():
    a8 = {s: .5 for s in (11,29,47)}
    raw = {s: .52 for s in a8}
    result = verdict_proposal(a8, .5, .008, raw, .5)
    assert result["verdict"] == "hòa" and result["raw_verdict"] == "thua"
    assert result["prior_dependent"] and result["approval_status"] == "pending_teacher"
    with pytest.raises(ValueError):
        verdict_proposal({11: .5}, .5, .008, {11: .5}, .5)


def test_qa23_all_keeps_flagged_columns_but_never_changes_main_or_raw():
    train, dev, full = _frames()
    # Force a non-approved allowlisted numeric proxy; QA23 runs only on train.
    train["avg_playback_speed"] = train.q_alpha_1
    before = train.copy(deep=True)
    main, audit, contract, _ = build_cd3_views(train, dev, full)
    all_ready, _, all_contract, _ = build_cd3_views(train, dev, full, feature_variant="qa23_all")
    assert "avg_playback_speed" in contract["qa23_flagged_columns"]
    assert "avg_playback_speed" not in contract["learner_columns"]
    assert "avg_playback_speed" in all_contract["learner_columns"]
    assert all_contract["held_out_columns"] == []
    assert all_contract["contract_hash"] != contract["contract_hash"]
    assert train.equals(before)
    assert all_contract["ready_learner_representation"].startswith("V1")
    assert not all_ready["train"].avg_playback_speed.equals(train.avg_playback_speed)


def test_feature_variant_is_hashed_and_must_match_runtime():
    main = ExperimentConfig.from_json(ROOT / "configs/cq_v4_review2310_l0_v1.json")
    all_cfg = replace(main, feature_variant="qa23_all")
    all_cfg.validate()
    assert protocol_hash(main) != protocol_hash(all_cfg)
    assert "feature_variant" not in main.to_dict()
    data = {"feature_contract": {"feature_view": "cd3", "feature_variant": "qa23_all"}}
    with pytest.raises(ValueError, match="feature_variant"):
        guard_feature_view(main, data)
    guard_feature_view(all_cfg, data)
    with pytest.raises(ValueError):
        replace(all_cfg, method_ids=("c02_stratified_random",)).validate()


def test_final_followups_add_S_and_separate_qa23_all_without_gate_recheck(tmp_path):
    base = ExperimentConfig.from_json(ROOT / "configs/cq_v4_review2310_l1_v1.json")
    result = followups(base, tmp_path / "qa23_processed")
    assert len(result) == 13
    assert result["S_final_ms42"].training_phases == ("P4",)
    assert result["qa23_all_ms42"].feature_variant == "qa23_all"
    assert result["qa23_all_ms42"].paths.processed_root != base.paths.processed_root
    with pytest.raises(ValueError):
        followups(base, base.paths.processed_root)


def test_endpoint_integration_A8_missing_class_fallback_and_E_CI(tmp_path):
    from scripts.prepare_cq_cd3_dev import prepare
    from scripts.compute_cq_v4_dev_endpoints import compute
    from skq_exp.experiments.runner import run_config
    from test_cq_review2310_pipeline import _sources
    raw, v4 = _sources(tmp_path)
    bundle, processed = tmp_path / "bundle", tmp_path / "processed"
    prepare(raw, v4, bundle, processed)
    base = ExperimentConfig.from_json(ROOT / "configs/cq_v4_review2310_lr_01_v1.json")
    config = replace(base, experiment_id="toy25", protocol_id="review25_toy",
        learner_params={"lr": {"standardize": True, "max_iter": 200, "C": 1., "solver": "lbfgs"}},
        paths=replace(base.paths, processed_root=processed, artifact_root=tmp_path / "artifacts"))
    assert all(r["status"] == "success" for r in run_config(config))
    report = compute(bundle, tmp_path / "artifacts", review_1025=True, n_bootstrap=20)["runs"][0]
    assert report["ul_decision"] == pytest.approx(report["ul_a8"])
    assert report["a8_undefined"] is False
    assert report["e_class_raw"]["pooled_ci95"]["ap_E"]["n_bootstrap"] == 20
    # Mutate synthetic cost metadata only to exercise the missing-source-class policy.
    for path in (tmp_path / "artifacts").rglob("cost.json"):
        cost = json.loads(path.read_text(encoding="utf-8"))
        cost["effective_class_prior"] = {"labels": [1,2], "probabilities": [.3,.7]}
        path.write_text(json.dumps(cost), encoding="utf-8")
    fallback = compute(bundle, tmp_path / "artifacts", review_1025=True, n_bootstrap=10)["runs"][0]
    assert fallback["a8_undefined"] and fallback["ul_a8"] is None
    assert fallback["ul_decision"] == fallback["ul_raw"]
    assert fallback["a8_prior_correction"]["pcc"] is None
    assert fallback["e_class_a8"] is None
