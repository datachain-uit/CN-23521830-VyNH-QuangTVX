"""Kiểm tra full config thật sự chứa toàn panel và các contract output tách biệt.

Test dùng mảng nhỏ nên không cần repo nặng. Nó ngăn việc vô tình đổi lệnh `full`
trở lại thành config chỉ có hai method như bản nháp ban đầu.
"""

from pathlib import Path

import numpy as np

from skq_exp.config import ExperimentConfig
from skq_exp.cli import _plan
from skq_exp.methods import METHOD_SPECS, allowed_learners_for, build_selector
from skq_exp.methods.proposed.query_losses import make_oof_query_losses
from skq_exp.methods.proposed.simplex_qp import solve_simplex_mean_match


ROOT = Path(__file__).resolve().parents[1]


def test_every_registered_method_has_an_explanatory_note():
    """Mỗi method phải tự mô tả vai trò/mức tái hiện để tránh ghi sai trong bài."""
    missing = [method_id for method_id, spec in METHOD_SPECS.items() if not spec.note.strip()]
    assert missing == []


def test_adult_full_config_has_20_rows_and_five_learners():
    config = ExperimentConfig.from_json(ROOT / "configs" / "pilot_adult_full_seed11.json")
    assert len(config.method_ids) == 20
    assert config.learner_ids == ("lr", "rf", "xgb", "cat", "mlp")
    assert set(config.method_ids) <= set(METHOD_SPECS)
    assert {
        METHOD_SPECS[item].output_kind
        for item in ("s_kip_tdbench", "s_mtt_tdbench", "s_tame_official")
    } == {"synthetic"}
    assert config.test_locked is True
    assert config.method_options["p04_skq_lrq_sq"]["parent_source"] == "n02_coretab_xgb_subset"
    assert config.method_options["p05_skq_lrq_mq"]["parent_source"] == "n02_coretab_xgb_subset"
    # Budget là biến thực nghiệm; KIP/MTT chỉ bị chặn bởi preflight tài nguyên có
    # đơn vị rõ ràng, không bởi trần số dòng tùy ý.
    assert "max_rows" not in config.method_options["s_kip_tdbench"]
    assert "max_rows" not in config.method_options["s_mtt_tdbench"]


def test_adult_full_v2_adds_deep_methods_and_plans_every_learner():
    config = ExperimentConfig.from_json(
        ROOT / "configs" / "pilot_adult_full_seed11_v2.json"
    )
    assert len(config.method_ids) == 22
    assert {"s_gm_tdbench", "s_datm_tdbench"} <= set(config.method_ids)
    assert config.protocol_id == "static_tabular_full_pilot_v2"
    assert allowed_learners_for(
        "c02_stratified_random", config.method_options["c02_stratified_random"]
    ) == ("lr", "rf", "xgb", "cat", "mlp")
    plan = _plan(config)
    assert plan["cells_total_per_dataset_seed"] == 110
    assert plan["cells_planned"] == 110
    assert plan["cells_na_contract"] == 0
    alias = ExperimentConfig.from_json(ROOT / "configs" / "pilot_adult_seed11.json")
    assert alias.method_ids == config.method_ids


def test_adult_full_v3_adds_predeclared_gonzalez_portfolio():
    config = ExperimentConfig.from_json(
        ROOT / "configs" / "pilot_adult_full_seed11_v3.json"
    )
    portfolio = {
        "p06_gonzalez_qp",
        "p07_skq_gonzalez",
        "p08_skq_gonzalez_lrq_sq",
        "p09_skq_gonzalez_lrq_mq",
    }
    assert len(config.method_ids) == 26
    assert portfolio <= set(config.method_ids)
    assert config.protocol_id == "static_tabular_full_pilot_v3_gonzalez_portfolio"
    assert config.test_locked is True
    assert config.method_options["p06_gonzalez_qp"]["candidate_multiplier"] == 1.0
    assert config.method_options["p07_skq_gonzalez"]["candidate_multiplier"] == 2.0
    assert all(
        config.method_options[method_id]["gonzalez_structure_groups"] == 128
        for method_id in portfolio
    )
    plan = _plan(config)
    assert plan["cells_total_per_dataset_seed"] == 130
    assert plan["cells_planned"] == 130
    assert plan["cells_na_contract"] == 0


def test_course_quality_v3_screens_full_portfolio_on_dev_before_temporal_freeze():
    config = ExperimentConfig.from_json(
        ROOT / "configs" / "pilot_course_quality_med_seed11_v3.json"
    )
    plan = _plan(config)
    assert len(config.method_ids) == 26
    assert config.dataset_ids == ("course_quality_med_v1",)
    assert plan["cells_planned"] == 130
    assert config.test_locked is True
    assert plan["evaluation_rows_total_per_dataset_seed"] == 130
    assert plan["evaluation_splits"] == ["dev"]


def test_adult_full_v4_adds_sharded_variants_without_changing_p06_p08():
    v3 = ExperimentConfig.from_json(
        ROOT / "configs" / "pilot_adult_full_seed11_v3.json"
    )
    v4 = ExperimentConfig.from_json(
        ROOT / "configs" / "pilot_adult_full_seed11_v4.json"
    )
    assert len(v4.method_ids) == 28
    assert {"p10_skq_mr_coretab_xgb", "p11_skq_mr_bdis"} <= set(v4.method_ids)
    assert v4.method_options["p06_gonzalez_qp"] == v3.method_options["p06_gonzalez_qp"]
    assert v4.method_options["p08_skq_gonzalez_lrq_sq"] == v3.method_options["p08_skq_gonzalez_lrq_sq"]
    assert v4.method_options["p10_skq_mr_coretab_xgb"]["shard_rows"] == 10_000
    plan = _plan(v4)
    assert plan["cells_planned"] == 140


def test_course_quality_temporal_v4_runs_28_methods_on_four_tests():
    config = ExperimentConfig.from_json(
        ROOT / "configs" / "pilot_course_quality_med_temporal_seed11_v4.json"
    )
    plan = _plan(config)
    assert len(config.method_ids) == 28
    assert plan["cells_planned"] == 140
    assert plan["evaluation_splits"] == [
        "test_phase1", "test_phase2", "test_phase3", "test_phase4",
    ]
    assert plan["evaluation_rows_total_per_dataset_seed"] == 560


def test_benchmark_selectors_return_exact_unique_indices(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 6))
    y = np.repeat([0, 1], 40)
    for method_id in ("n_gcoreset_benchmark", "n_leverage_benchmark"):
        selector = build_selector(method_id, 11, external_root=ROOT / "external" / "repos")
        result = selector.select(X, y, 0.1)
        assert result.status == "success"
        assert result.realized_rows == 8
        assert len(np.unique(result.indices)) == 8


def test_gonzalez_portfolio_keeps_final_selection_inside_candidate_pool():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(120, 7))
    y = np.repeat([0, 1], 60)
    gonzalez = build_selector(
        "n_gcoreset_benchmark", 11, external_root=ROOT / "external" / "repos"
    )
    pool_1x = gonzalez.select(X, y, 0.1)
    pool_2x = gonzalez.select(X, y, 0.2)
    mask_1x = np.isin(np.arange(len(y)), pool_1x.indices)
    mask_2x = np.isin(np.arange(len(y)), pool_2x.indices)

    p06 = build_selector("p06_gonzalez_qp", 11, n_components=32)
    weighted = p06.select(
        X, y, 0.1, parent_ids=np.zeros(len(y), dtype=np.int64),
        candidate_mask=mask_1x, row_ids=np.arange(len(y)),
    )
    assert weighted.status == "success"
    assert set(weighted.indices) == set(pool_1x.indices)
    assert not np.allclose(weighted.weights, weighted.weights[0])

    p07 = build_selector("p07_skq_gonzalez", 11, n_components=32)
    selected = p07.select(
        X, y, 0.1, parent_ids=np.zeros(len(y), dtype=np.int64),
        candidate_mask=mask_2x, row_ids=np.arange(len(y)),
    )
    assert selected.status == "success"
    assert selected.realized_rows == 12
    assert set(selected.indices) <= set(pool_2x.indices)


def test_oof_query_loss_is_deterministic_and_train_shaped():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(90, 5))
    y = np.repeat([0, 1, 2], 30)
    first, meta = make_oof_query_losses(X, y, ["lr"], seed=11, folds=3, max_threads=1)
    second, _ = make_oof_query_losses(X, y, ["lr"], seed=11, folds=3, max_threads=1)
    assert first.shape == (90, 1)
    assert np.allclose(first, second)
    assert meta["oof_folds"] == 3


def test_simplex_qp_keeps_constraints():
    rng = np.random.default_rng(3)
    selected = rng.normal(size=(12, 7))
    target = rng.normal(size=7)
    solution = solve_simplex_mean_match(selected, target)
    assert solution.converged
    assert np.isclose(solution.weights.sum(), 1.0)
    assert np.all(solution.weights >= -1e-12)
