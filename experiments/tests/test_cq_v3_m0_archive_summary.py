"""Kiểm tra trình tổng hợp archive M0 V3 và hàng rào dev-only."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from scripts.summarize_cq_v3_m0_archive import (
    EXPECTED_COMMIT,
    _forbidden_test_members,
    summarize,
)


def _archive(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            payload = json.dumps({"ok": True}).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _add_json(archive: tarfile.TarFile, name: str, value: object) -> None:
    payload = json.dumps(value).encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _add_text(archive: tarfile.TarFile, name: str, value: str) -> None:
    payload = value.encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _metric_path(experiment: str, method: str, budget: str, seed: int, learner: str, phase: int) -> str:
    return (
        f"artifacts/experiments/{experiment}/run/s1_screen/course_quality_v1/"
        f"{method}/{budget}/ss{seed}/{learner}/ms42/eval-dev_phase{phase}/metrics_overall.json"
    )


def _add_evaluation(
    archive: tarfile.TarFile, experiment: str, method: str, budget: str,
    seed: int, learner: str, phase: int, metrics: dict[str, float],
) -> None:
    path = _metric_path(experiment, method, budget, seed, learner, phase)
    _add_json(archive, path, metrics)
    rows = [
        "label,support,prevalence,precision,recall,specificity,f1,f2,mcc,roc_auc,pr_auc",
        "0,10,0.01,0.2,0.3,0.9,0.24,0.27,0.2,0.8,0.1",
        "1,300,0.30,0.6,0.7,0.8,0.65,0.68,0.5,0.85,0.6",
        "2,690,0.69,0.8,0.7,0.8,0.75,0.72,0.5,0.85,0.8",
    ]
    _add_text(archive, path.replace("metrics_overall.json", "metrics_per_label.csv"), "\n".join(rows) + "\n")


def test_forbidden_test_members_are_precise() -> None:
    names = [
        "m0_docs/pytest_m0_contracts.xml",
        "artifacts/eval-dev_phase1/metrics_overall.json",
        "source/model_ready/P1/test.parquet",
        "artifacts/eval-test/metrics_overall.json",
        "artifacts/private/test_predictions.parquet",
    ]
    assert _forbidden_test_members(names) == [
        "artifacts/eval-test/metrics_overall.json",
        "artifacts/private/test_predictions.parquet",
        "source/model_ready/P1/test.parquet",
    ]


def test_tar_fixture_does_not_need_extraction(tmp_path: Path) -> None:
    path = tmp_path / "fixture.tar.gz"
    _archive(path, ["m0_docs/pytest_m0_contracts.xml"])
    assert path.is_file()
    assert not (tmp_path / "m0_docs").exists()


def test_complete_dev_archive_is_ready_only_for_conditional_go(tmp_path: Path) -> None:
    path = tmp_path / "complete.tar.gz"
    checks = {
        name: {"status": status}
        for name, status in {
            "D01_size_distribution_split": "PASS",
            "D02_id_only_probe": "PASS",
            "D03_missingness_probe": "PASS",
            "D04_shuffled_labels": "PASS",
            "D05_majority_baseline": "PASS",
            "D06_cutoff_manual_20": "PASS",
            "D07_weight_contract": "PASS_CODE",
            "D09_unit_label": "PASS",
            "D10_no_label_in_X": "PASS",
            "D11_course_recompute": "PASS_GENERATED",
            "D12_protocol_constants": "PASS",
            "course_root_contract": "BLOCKED",
            "split_protocol_contract": "BLOCKED_APPROVAL",
        }.items()
    }
    metrics = {
        "accuracy": 0.7, "balanced_accuracy": 0.6, "mcc": 0.5,
        "f1_macro": 0.6, "worst_class_f1": 0.2, "worst_class_recall": 0.3,
        "roc_auc_macro": 0.8, "pr_auc_macro": 0.5, "log_loss": 0.7,
        "brier_multiclass": 0.4, "ece_15": 0.1,
    }
    with tarfile.open(path, "w:gz") as archive:
        status = {
            "status": "success_conditional_go_evidence_ready",
            "repository_commit": EXPECTED_COMMIT,
            "test_files_permitted": False,
            "test_metrics_opened": False,
            "fulltrain_a_status": "blocked_not_in_source_bundle",
            "steps": {"00": {"status": "success"}},
        }
        _add_json(archive, "cq_v3_m0_execution_status.json", status)
        _add_text(archive, "repository_commit_cq_v3_m0.txt", EXPECTED_COMMIT + "\n")
        _add_json(archive, "m0_docs/cq_v3/m0_probes_dev_summary.json", {"D04": {"status": "PASS"}})
        _add_json(archive, "m0_docs/cq_v3/m0_data_card_provisional.json", {
            "test_data_packaged": False, "checks": checks,
        })
        _add_json(archive, "m0_docs/cq_v3/m0_headroom_decision_cq_v3.json", {
            "evidence_split": "dev", "test_data_read": False, "rho_star": 0.005,
            "fallback_to_smallest": True,
            "ratios": {str(value): {} for value in (0.005, 0.01, 0.02, 0.05)},
        })
        for policy in ("balanced", "natural"):
            experiment = f"cq_v3_m0_fulltrain_b_dev_{policy}_v1"
            for learner in ("lr", "xgb"):
                for phase in range(1, 5):
                    _add_evaluation(
                        archive, experiment, "c00_full_train", "b1p0000", 0,
                        learner, phase, metrics,
                    )
        experiment = "cq_v3_m0_fulltrain_b_dev_natural_xgb_v1"
        for phase in range(1, 5):
            _add_evaluation(
                archive, experiment, "c00_full_train", "b1p0000", 0,
                "xgb", phase, metrics,
            )
        for token in ("rho005", "rho01", "rho02", "rho05"):
            experiment = f"cq_v3_m0_headroom_c02_{token}_natural_v1"
            for seed in (11, 29, 47):
                for phase in range(1, 5):
                    _add_evaluation(
                        archive, experiment, "c02_stratified_random", "b0p0050",
                        seed, "xgb", phase, metrics,
                    )
    result = summarize(path)
    assert result["request_status"] == "READY_TO_REQUEST_CONDITIONAL_GO"
    assert result["full_go"] is False
    assert result["privacy"]["pass"] is True
    assert result["evaluation_rows"] == 68
    assert result["per_label_rows"] == 204
    assert len(result["per_label_summary"]) == 12


def test_failed_diagnostic_archive_is_summarized_not_rejected(tmp_path: Path) -> None:
    path = tmp_path / "failed.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _add_json(archive, "cq_v3_m0_execution_status.json", {
            "status": "failed_diagnostic_archive_created",
            "test_files_permitted": False,
            "test_metrics_opened": False,
            "steps": {"03": {"status": "failed", "error": "bad bundle"}},
        })
    result = summarize(path)
    assert result["request_status"] == "NOT_READY"
    assert result["step_failures"]["03"]["status"] == "failed"
