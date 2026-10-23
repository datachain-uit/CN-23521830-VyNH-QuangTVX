"""File này tạo và kiểm tra gói ZIP nhẹ để lưu bằng chứng và chạy lại.

Gói ``replay-lite`` giữ code, config, môi trường, checksum, metric và canonical
selection/synthetic payload nhưng bỏ raw/processed arrays, model và probability.
Nhờ vậy ta có thể tải lại dữ liệu rồi retrain mà không lưu lặp nhiều GB.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable


SMALL_RUN_FILES = {
    "manifest.json",
    "method_artifact_ref.json",
    "diagnostics.json",
    "timings.json",
    "evaluation_failure.json",
    "metrics_overall.json",
    "metrics_all.json",
    "metrics_per_label.csv",
    "confusion_matrix.csv",
    "calibration.csv",
    "cost.json",
    "retention.json",
}


def storage_report(artifact_root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Trả bảng dung lượng theo nhóm để thấy file nào đang làm phình ổ đĩa."""
    root = Path(artifact_root)
    groups = {
        "learner_runs": root / "experiments" / experiment_id,
        "method_artifacts": root / "method_artifacts" / experiment_id,
        "structures": root / "structures" / experiment_id,
        "ledger": root / experiment_id,
    }
    summary = {}
    all_files: list[Path] = []
    for name, base in groups.items():
        files = [path for path in base.rglob("*") if path.is_file()] if base.exists() else []
        all_files.extend(files)
        summary[name] = {
            "files": len(files),
            "bytes": int(sum(path.stat().st_size for path in files)),
        }
    largest = sorted(set(all_files), key=lambda path: path.stat().st_size, reverse=True)[:20]
    return {
        "experiment_id": experiment_id,
        "artifact_root": str(root.resolve()),
        "groups": summary,
        "total_bytes": int(sum(item["bytes"] for item in summary.values())),
        "largest_files": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
            for path in largest
        ],
    }


def pack_experiment(
    *,
    project_root: str | Path,
    artifact_root: str | Path,
    experiment_id: str,
    output: str | Path,
    profile: str = "replay-lite",
    max_archive_mb: float = 512.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Đóng gói experiment có manifest checksum và ngưỡng chống tràn đĩa."""
    if profile not in {"replay-lite", "results-only"}:
        raise ValueError("profile phải là replay-lite hoặc results-only")
    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    output = Path(output).resolve()
    files = _archive_files(project_root, artifact_root, experiment_id, profile)
    raw_bytes = int(sum(path.stat().st_size for path, _ in files))
    max_bytes = int(max_archive_mb * 1024 ** 2)
    plan = {
        "schema_version": 3,
        "profile": profile,
        "experiment_id": experiment_id,
        "file_count": len(files),
        "raw_bytes": raw_bytes,
        "max_archive_bytes": max_bytes,
        "output": str(output),
        "excluded": [
            "raw data", "processed arrays", "external repo checkouts",
            "fitted models", "per-run probabilities/predictions",
        ],
    }
    if raw_bytes > max_bytes:
        raise RuntimeError(
            f"Gói dự kiến {raw_bytes / 1024**2:.1f} MiB vượt giới hạn "
            f"{max_archive_mb:.1f} MiB; dùng results-only hoặc tăng giới hạn có chủ đích"
        )
    if dry_run:
        return plan
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, arcname in files:
            archive.write(source, arcname)
            entries.append({
                "path": arcname,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            })
        environment = _environment_snapshot(project_root)
        archive.writestr(
            "environment.json",
            json.dumps(environment, ensure_ascii=False, indent=2),
        )
        archive.writestr("REPLAY.md", _replay_instructions(experiment_id, profile))
        archive_manifest = {
            **plan,
            "created_unix": time.time(),
            "entries": entries,
            "environment_sha256": hashlib.sha256(
                json.dumps(environment, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
        archive.writestr(
            "archive_manifest.json",
            json.dumps(archive_manifest, ensure_ascii=False, indent=2),
        )
    plan["zip_bytes"] = output.stat().st_size
    plan["sha256"] = _sha256(output)
    return plan


def verify_archive(path: str | Path) -> dict[str, Any]:
    """Kiểm checksum từng entry; không giải nén và không ghi ra đĩa."""
    path = Path(path)
    failures = []
    with zipfile.ZipFile(path) as archive:
        archive.testzip()
        manifest = json.loads(archive.read("archive_manifest.json"))
        for entry in manifest.get("entries", []):
            payload = archive.read(entry["path"])
            actual = hashlib.sha256(payload).hexdigest()
            if actual != entry["sha256"] or len(payload) != int(entry["bytes"]):
                failures.append(entry["path"])
    return {
        "archive": str(path.resolve()),
        "valid": not failures,
        "failed_entries": failures,
        "entries_checked": len(manifest.get("entries", [])),
        "profile": manifest.get("profile"),
        "experiment_id": manifest.get("experiment_id"),
    }


def _archive_files(
    project_root: Path,
    artifact_root: Path,
    experiment_id: str,
    profile: str,
) -> list[tuple[Path, str]]:
    collected: dict[str, Path] = {}

    def add(source: Path, arcname: str) -> None:
        if source.is_file():
            collected.setdefault(arcname.replace("\\", "/"), source)

    for name in ["pyproject.toml", "README.md", "GUIDELINE_GITHUB_KAGGLE.md"]:
        add(project_root / name, f"project/{name}")
    for directory in ["src", "configs", "requirements", "scripts", "docs"]:
        base = project_root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                add(path, f"project/{path.relative_to(project_root).as_posix()}")
    add(
        project_root / "external" / "official_repos.lock.json",
        "project/external/official_repos.lock.json",
    )

    run_root = artifact_root / "experiments" / experiment_id
    if run_root.exists():
        for path in run_root.rglob("*"):
            if path.is_file() and path.name in SMALL_RUN_FILES:
                add(path, f"project/artifacts/{path.relative_to(artifact_root).as_posix()}")
    for path in [
        artifact_root / experiment_id / "run_ledger.json",
        artifact_root / "freeze" / "freeze_manifest.json",
    ]:
        add(path, f"project/artifacts/{path.relative_to(artifact_root).as_posix()}")
    aggregate = artifact_root / "aggregate"
    if aggregate.exists():
        for path in aggregate.glob("*.csv"):
            add(path, f"project/artifacts/{path.relative_to(artifact_root).as_posix()}")
    if profile == "replay-lite":
        for category in ["method_artifacts", "structures"]:
            base = artifact_root / category / experiment_id
            if base.exists():
                for path in base.rglob("*"):
                    if path.is_file():
                        add(path, f"project/artifacts/{path.relative_to(artifact_root).as_posix()}")

    dataset_ids = _dataset_ids(run_root)
    for dataset_id in dataset_ids:
        processed = project_root / "data" / "processed" / dataset_id
        for filename in [
            "preprocessing_manifest.json", "split_manifest.json", "schema.json",
        ]:
            add(
                processed / filename,
                f"provenance/data_manifests/{dataset_id}/{filename}",
            )
    return [(source, arcname) for arcname, source in sorted(collected.items())]


def _dataset_ids(run_root: Path) -> set[str]:
    values = set()
    if not run_root.exists():
        return values
    for manifest_path in run_root.rglob("manifest.json"):
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_id")
            if value:
                values.add(str(value))
        except (OSError, ValueError, TypeError):
            continue
    return values


def _environment_snapshot(project_root: Path) -> dict[str, Any]:
    packages = sorted(
        ({"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
         for dist in importlib.metadata.distributions()),
        key=lambda item: item["name"].lower(),
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git": _git_state(project_root),
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(arguments: Iterable[str]) -> str | None:
        try:
            completed = subprocess.run(
                list(arguments), cwd=project_root, check=True,
                capture_output=True, text=True, timeout=15,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "root": run(["git", "rev-parse", "--show-toplevel"]),
        "status_project": run(["git", "status", "--short", "--", "."]),
    }


def _replay_instructions(experiment_id: str, profile: str) -> str:
    return f"""# Khôi phục experiment {experiment_id}

Profile: `{profile}`.

1. Giải nén ZIP và vào thư mục `project`.
2. Tạo môi trường Python phù hợp với `environment.json`, rồi cài `pip install -e .`.
3. Tải raw data từ nguồn chính thức và chạy lại `skq prepare` để tạo processed arrays.
4. Kiểm tra preprocessing manifest mới khớp checksum trong `provenance/data_manifests`.
5. Dùng đúng config đã đóng gói và chạy `skq run --reuse-method-artifacts --resume ...`.
6. Chạy `skq aggregate --artifact-root artifacts` để dựng lại bảng tổng hợp.

Gói không chứa raw data, processed arrays, external repository checkout, fitted
model hay probability. `replay-lite` giữ canonical selection/weights/structure và
synthetic payload; `results-only` chỉ giữ bằng chứng và bảng kết quả, không đủ để
retrain mà không chạy lại selector/generator.
"""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
