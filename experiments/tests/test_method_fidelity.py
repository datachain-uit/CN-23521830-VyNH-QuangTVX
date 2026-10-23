"""Kiểm tra metadata khoa học không bị đổi thành claim bằng ghi chú tự do."""

from skq_exp.methods.registry import METHOD_SPECS


def test_coretab_native_is_explicitly_binary_only() -> None:
    for method_id in ("n01_coretab_dt_subset", "n02_coretab_xgb_subset"):
        spec = METHOD_SPECS[method_id]
        assert spec.task_contract == "binary_only"
        assert spec.fidelity_tier == "official_source_adapter"
        assert spec.budget_contract == "native_realized"


def test_historical_p_variants_are_not_confirmatory() -> None:
    for method_id in (
        "p01_skq_coretab_dt", "p02_skq_coretab_xgb", "p03_skq_bdis_filtered",
        "p04_skq_lrq_sq", "p05_skq_lrq_mq", "p06_gonzalez_qp",
        "p07_skq_gonzalez", "p08_skq_gonzalez_lrq_sq",
        "p09_skq_gonzalez_lrq_mq", "p10_skq_mr_coretab_xgb",
        "p11_skq_mr_bdis",
    ):
        assert METHOD_SPECS[method_id].confirmatory_allowed is False


def test_pending_coretab_cell_candidate_is_runnable_only_as_exploratory() -> None:
    spec = METHOD_SPECS["x_coretab_mc_cell"]
    assert spec.status == "implemented"
    assert spec.role == "experimental_candidate"
    assert spec.evidence_role == "exploratory"
    assert spec.confirmatory_allowed is False
    assert spec.task_contract == "task_agnostic"
    assert spec.needs_parent is True


def test_three_reviewed_skq_pc_candidates_are_registered_for_dev_selection() -> None:
    for method_id in ("skq_pc", "skq_pc_lrq_sq", "skq_pc_sqrt"):
        spec = METHOD_SPECS[method_id]
        assert spec.status == "implemented"
        assert spec.role == "proposed"
        assert spec.evidence_role == "dev_selection"
        assert spec.confirmatory_allowed is True
        assert spec.max_dev_configs == 1


def test_skq_pc_r_is_review_1022_ablation_not_candidate() -> None:
    spec = METHOD_SPECS["skq_pc_r"]
    assert spec.status == "implemented"
    assert spec.role == "ablation"
    assert spec.evidence_role == "appendix"


def test_c01_c03_are_controls_not_proposed_methods() -> None:
    for method_id in ("c01_uniform_random", "c03_phase_class_random"):
        spec = METHOD_SPECS[method_id]
        assert spec.role == "control"
        assert spec.fidelity_tier == "local_reimplementation"
        assert spec.confirmatory_allowed is True


def test_coretab_ova_has_adapted_multiclass_contract() -> None:
    spec = METHOD_SPECS["a_coretab_xgb_ova"]
    assert spec.role == "adaptation"
    assert spec.task_contract == "multiclass"
    assert "adapted" in spec.display_name.lower()
    assert METHOD_SPECS["n02_coretab_xgb_subset"].task_contract == "binary_only"


def test_review_1012_modern_baselines_are_registered_with_honest_labels() -> None:
    expected = {
        "b_moderate_ds_adapted": ("baseline", "official_source_adapter"),
        "b_ccs_oof_xgb_adapted": ("baseline", "official_source_adapter"),
        "b_dataoob_official": ("baseline", "official_source_with_declared_patches"),
        "b_kernel_thinning_classwise": ("baseline", "official_source_adapter"),
    }
    for method_id, (role, fidelity) in expected.items():
        spec = METHOD_SPECS[method_id]
        assert spec.status == "implemented"
        assert spec.role == role
        assert spec.fidelity_tier == fidelity
        assert spec.confirmatory_allowed is True
        assert spec.evidence_role == "confirmatory"
        assert spec.budget_contract == "exact"

    assert "adapted" in METHOD_SPECS["b_moderate_ds_adapted"].display_name.lower()
    assert "adapted" in METHOD_SPECS["b_ccs_oof_xgb_adapted"].display_name.lower()
    assert "adapted" in METHOD_SPECS["b_kernel_thinning_classwise"].display_name.lower()

