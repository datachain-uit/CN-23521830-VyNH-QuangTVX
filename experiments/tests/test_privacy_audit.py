"""Privacy audit phải giữ finding riêng theo tracked/history/archive."""

from __future__ import annotations

import zipfile
from pathlib import Path

from skq_exp.reports import privacy as privacy_module


def test_public_path_allowlist_and_private_artifacts() -> None:
    assert privacy_module.classify_public_path("data/raw/.gitkeep") is None
    assert privacy_module.classify_public_path("src/skq_exp/config.py") is None
    assert privacy_module.classify_public_path("data/processed/cq/X_train.npy")["kind"] == "private_data_path"
    assert privacy_module.classify_public_path("selected_indices.npy")["kind"] == "private_artifact_type"


def test_audit_keeps_tracked_history_and_archive_findings_separate(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "safe.py").write_text("print('safe')", encoding="utf-8")
    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("project/predictions.csv", "row_id,prediction\n1,0\n")

    def _git_lines(_root, command):
        if command[1:3] == ["ls-files"]:
            return ["safe.py"]
        if command[1:3] == ["rev-list", "--objects"]:
            return ["deadbeef data/raw/cq/train.csv"]
        if command[1:3] == ["rev-parse", "HEAD"]:
            return ["abc123"]
        raise AssertionError(command)

    monkeypatch.setattr(privacy_module, "_git_lines", _git_lines)
    evidence = privacy_module.audit_public_release(project, archives=[archive_path])
    assert evidence["status"] == "fail"
    assert evidence["tracked_findings"] == []
    assert evidence["history_findings"][0]["kind"] == "private_data_path"
    assert evidence["archive_findings"][0]["kind"] == "row_level_artifact"


def test_secret_is_reported_without_copying_secret_value(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    # Ghép runtime để source test không tự chứa chuỗi giống credential thật.
    (project / "config.txt").write_text(
        "KAGGLE" + "_KEY=" + "abcdefghijkl" + "mnop1234", encoding="utf-8",
    )

    def _git_lines(_root, command):
        if command[1:3] == ["ls-files"]:
            return ["config.txt"]
        if command[1:3] == ["rev-list", "--objects"]:
            return []
        if command[1:3] == ["rev-parse", "HEAD"]:
            return ["abc123"]
        raise AssertionError(command)

    monkeypatch.setattr(privacy_module, "_git_lines", _git_lines)
    evidence = privacy_module.audit_public_release(project)
    finding = evidence["tracked_findings"][0]
    assert finding["kind"] == "kaggle_key"
    assert "abcdefghijklmnop1234" not in str(finding)
