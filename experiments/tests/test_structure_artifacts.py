"""Structure artifact phải mang đủ identity và không được replay sang split khác."""

from pathlib import Path

import numpy as np

from skq_exp.artifacts import atomic_json
from skq_exp.config import ExperimentConfig, ProjectPaths
from skq_exp.experiments.runner import _resolve_structure, _save_structure
from skq_exp.methods.result import SelectionResult


def _config(tmp_path: Path) -> ExperimentConfig:
    processed = tmp_path / "processed"
    dataset = processed / "adult_uci2_v1"
    dataset.mkdir(parents=True)
    atomic_json(dataset / "split_manifest.json", {"split": "registered"})
    return ExperimentConfig(
        schema_version=3,
        experiment_id="structure_test",
        protocol_id="static_tabular_v1",
        stage_id="s1_screen",
        dataset_ids=("adult_uci2_v1",),
        method_ids=("n01_coretab_dt_subset",),
        learner_ids=("lr",),
        budget_ratio=0.05,
        split_seed=20261002,
        selector_seeds=(11,),
        model_seed=42,
        paths=ProjectPaths(
            raw_root=tmp_path / "raw",
            processed_root=processed,
            artifact_root=tmp_path / "artifacts",
            external_root=tmp_path / "external" / "repos",
        ),
    )


def test_structure_roundtrip_checks_row_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    row_ids = np.asarray([101, 102, 103, 104], dtype=np.int64)
    structure = {
        "row_ids": row_ids,
        "parent_ids": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "candidate_mask": np.asarray([True, False, True, True]),
        "_producer_total_seconds": np.asarray(1.5),
    }
    result = SelectionResult(
        indices=np.asarray([0, 2]), weights=np.ones(2), requested_rows=2,
        realized_rows=2, budget_mode="native_realized",
        method_id="n01_coretab_dt_subset",
        diagnostics={"upstream_commit": "locked-commit"},
    )
    _save_structure(
        config, "adult_uci2_v1", "n01_coretab_dt_subset", 11,
        structure, result, "dataset-fingerprint",
    )
    loaded = _resolve_structure(
        {"parent_source": "n01_coretab_dt_subset"}, 11, {}, config,
        "adult_uci2_v1", row_ids, "dataset-fingerprint",
    )
    assert not isinstance(loaded, str)
    assert np.array_equal(loaded["parent_ids"], structure["parent_ids"])
    rejected = _resolve_structure(
        {"parent_source": "n01_coretab_dt_subset"}, 11, {}, config,
        "adult_uci2_v1", row_ids + 1, "dataset-fingerprint",
    )
    assert isinstance(rejected, str)
    assert "row IDs" in rejected
