"""Freeze manifest phải khóa đúng config, commit và processed snapshot."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from skq_exp.config import ExperimentConfig, ProjectPaths
from skq_exp.experiments import freeze as freeze_module
from skq_exp.experiments.provenance import protocol_hash
from skq_exp.experiments.runner import _method_artifact_identity


def test_create_transferred_freeze_manifest(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).parents[1]
    source = ExperimentConfig.from_json(
        root / "configs" / "pilot_course_quality_med_temporal_seed11.json"
    )
    processed = tmp_path / "processed"
    manifest = processed / "course_quality_med_v1" / "preprocessing_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"snapshot": "cq-test"}', encoding="utf-8")
    config = replace(
        source,
        paths=ProjectPaths(
            raw_root=tmp_path / "raw",
            processed_root=processed,
            artifact_root=tmp_path / "artifacts",
            external_root=tmp_path / "external",
        ),
    )
    monkeypatch.setattr(freeze_module, "_git_commit", lambda _root: "abc123")

    output = freeze_module.create_freeze_manifest(
        config,
        base_winner_method_id="p02_skq_coretab_xgb",
        base_winner_structure_source="n02_coretab_xgb_subset",
        published_reference="n02_coretab_xgb_subset",
        proposed_winner="p05_skq_lrq_mq",
        selection_basis="transferred_from_adult_before_cq_test",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["code_commit"] == "abc123"
    assert payload["selection_basis"] == "transferred_from_adult_before_cq_test"
    assert payload["dataset_fingerprints"]["course_quality_med_v1"]
    assert payload["method_roles"]["proposed_winner"] == "p05_skq_lrq_mq"
    assert payload["allowed_parent_sources"] == ["n02_coretab_xgb_subset"]
    identity = _method_artifact_identity(
        config, "course_quality_med_v1", "p02_skq_coretab_xgb", 11, "fingerprint",
    )
    assert identity["protocol_hash"] == protocol_hash(config)


def test_freeze_accepts_predeclared_gonzalez_parent(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).parents[1]
    source = ExperimentConfig.from_json(
        root / "configs" / "pilot_course_quality_med_temporal_seed11_v4.json"
    )
    processed = tmp_path / "processed"
    manifest = processed / "course_quality_med_v1" / "preprocessing_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"snapshot": "cq-test"}', encoding="utf-8")
    config = replace(
        source,
        paths=ProjectPaths(
            raw_root=tmp_path / "raw",
            processed_root=processed,
            artifact_root=tmp_path / "artifacts",
            external_root=tmp_path / "external",
        ),
    )
    monkeypatch.setattr(freeze_module, "_git_commit", lambda _root: "abc123")
    output = freeze_module.create_freeze_manifest(
        config,
        base_winner_method_id="p02_skq_coretab_xgb",
        base_winner_structure_source="n02_coretab_xgb_subset",
        published_reference="n02_coretab_xgb_subset",
        proposed_winner="p08_skq_gonzalez_lrq_sq",
        selection_basis="transferred_full_adult_portfolio_before_cq_test",
        allowed_parent_sources=("gonzalez_pool_2x",),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["allowed_parent_sources"] == [
        "n02_coretab_xgb_subset", "gonzalez_pool_2x",
    ]
