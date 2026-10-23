"""Contracts for dev-only A2/A4 and source-quota matched C03.

Synthetic data only. A4 is an index-preserving weight ablation, not evidence
that selection without QP has the measured cost of the derived reweight step.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skq_exp.config import ExperimentConfig
from skq_exp.experiments.runner import _produce_or_block
from skq_exp.methods import build_selector
from skq_exp.methods.proposed.cq_ablations import (
    A2, A4, C03_MATCHED, ClassOnlySKQPC, derive_phase_class_selection,
    resolve_ablation_options,
)


def pool():
    X = np.random.default_rng(209).normal(size=(160, 5)).astype(np.float32)
    y = np.tile(np.r_[np.zeros(32, dtype=int), np.ones(8, dtype=int)], 4)
    phase = np.repeat(np.arange(4), 40)
    return X, y, phase, np.arange(160)


def options(source):
    return {"source_method": source, "n_components": 16, "max_cells_per_stratum": 4,
            "allocation_policy": "sqrt_floor" if source.endswith("sqrt") else "proportional", "allocation_floor": 1}


@pytest.mark.parametrize("source", ["skq_pc", "skq_pc_sqrt", "skq_pc_lrq_sq"])
def test_a2_is_exact_class_only_engine_with_same_source_options(source):
    X, y, phase, rows = pool()
    extra = {"query_features": np.linspace(-1, 1, len(y))[:, None]} if "lrq" in source else {}
    actual = build_selector(A2, 11, options=options(source)).select(X, y, .2, phases=phase, row_ids=rows, **extra)
    expected = build_selector(source, 11, options=options(source)).select(X, y, .2, phases=np.full(len(y), "all_phases", dtype=object), row_ids=rows, **extra)
    assert actual.status == expected.status == "success"
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.weights, expected.weights)
    assert actual.method_id == A2 and actual.diagnostics["n_strata"] == 2
    assert actual.diagnostics["stratification"] == "class_only"
    assert actual.diagnostics["phase_mass_preservation_required"] is False
    for label in np.unique(y):
        assert np.isclose(actual.weights[y[actual.indices] == label].sum(), np.sum(y == label))


@pytest.mark.parametrize("source", ["skq_pc", "skq_pc_sqrt", "skq_pc_lrq_sq"])
def test_a4_and_c03_preserve_exact_source_quota_mass_and_source_bytes(source, monkeypatch):
    X, y, phase, rows = pool()
    extra = {"query_features": np.linspace(-1, 1, len(y))[:, None]} if "lrq" in source else {}
    base = build_selector(source, 11, options=options(source)).select(X, y, .2, phases=phase, row_ids=rows, **extra)
    assert base.status == "success"
    original_indices, original_weights = base.indices.copy(), base.weights.copy()
    import skq_exp.methods.proposed.skq_pc as engine
    monkeypatch.setattr(engine, "solve_simplex_mean_match", lambda *a, **k: pytest.fail("derived method must not call QP"))
    for method in (A4, C03_MATCHED):
        actual = derive_phase_class_selection(method, base, y, phase, .2, seed=29)
        assert actual.status == "success" and len(np.unique(actual.indices)) == 32
        assert actual.diagnostics["source_quota_matched"] is True
        assert actual.diagnostics["qp_applied_in_derived_step"] is False
        assert actual.diagnostics["source_qp_cost_already_incurred"] is True
        assert actual.timings["total"] >= actual.timings["source_selection"]
        for p in np.unique(phase):
            for label in np.unique(y):
                population = (phase == p) & (y == label)
                selected = population[actual.indices]
                assert selected.sum() == population[base.indices].sum()
                assert np.isclose(actual.weights[selected].sum(), population.sum())
                assert len(np.unique(actual.weights[selected])) == 1
        if method == A4:
            np.testing.assert_array_equal(actual.indices, base.indices)
        else:
            repeat = derive_phase_class_selection(method, base, y, phase, .2, seed=29)
            np.testing.assert_array_equal(actual.indices, repeat.indices)
    np.testing.assert_array_equal(base.indices, original_indices)
    np.testing.assert_array_equal(base.weights, original_weights)


def test_ablation_options_fail_closed_instead_of_using_default_winner():
    with pytest.raises(ValueError, match="explicit"):
        resolve_ablation_options(A2, {})
    with pytest.raises(ValueError, match="only source_method"):
        resolve_ablation_options(A2, {A2: {"source_method": "skq_pc", "n_components": 999}, "skq_pc": {}})
    with pytest.raises(ValueError, match="declared"):
        resolve_ablation_options(A2, {A2: {"source_method": "skq_pc"}})
    table = {A2: {"source_method": "skq_pc_sqrt"}, "skq_pc_sqrt": {"allocation_floor": 7}}
    assert resolve_ablation_options(A2, table)["allocation_floor"] == 7


def config(tmp_path, source="skq_pc"):
    base = ExperimentConfig.from_json(Path(__file__).parents[1] / "configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json")
    return replace(base, experiment_id="toy_ablation", budget_ratio=.2, method_ids=(source, A2, A4, C03_MATCHED),
        method_options={source: options(source), **{method: {"source_method": source} for method in (A2, A4, C03_MATCHED)}},
        paths=replace(base.paths, processed_root=tmp_path / "processed", artifact_root=tmp_path / "artifacts"))


@pytest.mark.parametrize("source", ["skq_pc", "skq_pc_sqrt", "skq_pc_lrq_sq"])
def test_runner_uses_declared_source_and_train_only_query_contract(tmp_path, source):
    X, y, phase, rows = pool()
    data = {"X_train": X, "y_train": y, "phase_train": phase, "row_ids_train": rows}
    cfg, cache = config(tmp_path, source), {}
    queries = {(("lr",), 11): (np.linspace(-1, 1, len(y))[:, None], {"seconds": .125, "source": "toy"})}
    for method in cfg.method_ids:
        result = _produce_or_block(method, 11, cfg, data, "course_quality_v4", cache, {}, queries, "toy", None)
        assert result.status == "success", result.diagnostics
        cache[(method, 11)] = result
    assert cache[(A4, 11)].diagnostics["same_indices"] is True
    if "lrq" in source:
        assert cache[(A2, 11)].diagnostics["query_metadata"]["query_contract"] == "OOF-LR train-only"
        assert cache[(source, 11)].timings["query"] == .125
        assert cache[(A2, 11)].timings["query"] == .125
        assert cache[(A2, 11)].diagnostics["query_cost_shared"] is True
        assert cache[(A2, 11)].diagnostics["query_cache_hit"] is True


@pytest.mark.parametrize("method", [A2, A4, C03_MATCHED])
def test_prepared_methods_do_not_open_test_or_fall_back(tmp_path, method):
    X, y, phase, rows = pool()
    data = {"X_train": X, "y_train": y, "phase_train": phase, "row_ids_train": rows}
    cfg = config(tmp_path)
    locked = _produce_or_block(method, 11, replace(cfg, test_locked=False), data, "course_quality_v4", {}, {}, {}, "toy", None)
    assert locked.status == "gate_locked"
    if method != A2:
        missing = _produce_or_block(method, 11, cfg, data, "course_quality_v4", {}, {}, {}, "toy", None)
        assert missing.status == "blocked" and "source" in missing.diagnostics["reason"]


def test_missing_phase_budget_and_source_coverage_are_not_silently_repaired():
    X, y, phase, rows = pool()
    assert ClassOnlySKQPC(11, options("skq_pc")).select(X, y, .2).status == "blocked"
    base = build_selector("skq_pc", 11, options=options("skq_pc")).select(X, y, .2, phases=phase, row_ids=rows)
    assert derive_phase_class_selection(A4, base, y, phase, .3, seed=11).status == "blocked"
    assert derive_phase_class_selection(A4, base, y, None, .2, seed=11).status == "blocked"


@pytest.mark.parametrize("source", ["skq_pc", "skq_pc_sqrt", "skq_pc_lrq_sq"])
def test_prepared_templates_validate_without_selecting_a_winner(source):
    path = Path(__file__).parents[1] / f"configs/cq_v4_dev_ablations_from_{source}_v1.json"
    cfg = ExperimentConfig.from_json(path)
    cfg.validate()
    assert cfg.test_locked and cfg.stage_id == "s1_screen"
    assert cfg.method_ids == (source, A2, A4, C03_MATCHED)
    assert cfg.selector_seeds == (11, 29, 47)
    for method in (A2, A4, C03_MATCHED):
        assert resolve_ablation_options(method, cfg.method_options)["source_method"] == source


def test_runner_saves_four_phase_artifacts_for_all_prepared_methods(tmp_path, monkeypatch):
    from skq_exp.experiments import runner
    X, y, phase, rows = pool()
    data = {"X_train": X, "y_train": y, "phase_train": phase, "row_ids_train": rows,
            "X_dev": X[:40], "y_dev": y[:40]}
    for i in range(1, 5):
        data[f"X_dev_phase{i}"] = X[:40]
        data[f"y_dev_phase{i}"] = y[:40]
    # Short synthetic identities avoid Windows MAX_PATH; real config names stay unchanged.
    cfg = replace(config(tmp_path, "skq_pc_sqrt"), experiment_id="a", protocol_id="toy",
                  selector_seeds=(11,), learner_ids=("lr",),
                  paths=replace(config(tmp_path).paths, artifact_root=tmp_path.parent / "a"),
                  learner_params={"lr": {"max_iter": 200}}, early_stopping={"enabled": False})
    monkeypatch.setattr(runner, "load_processed", lambda *a, **k: data)
    monkeypatch.setattr(runner, "dataset_fingerprint", lambda *a, **k: "synthetic-only")
    ledger = runner.run_config(cfg)
    assert len(ledger) == 16
    assert all(row["status"] == "success" for row in ledger)
    assert {row["method_id"] for row in ledger} == set(cfg.method_ids)
    assert len(list(cfg.paths.artifact_root.rglob("metrics_overall.json"))) == 16
    assert len(list(cfg.paths.artifact_root.rglob("metrics_per_label.csv"))) == 16
