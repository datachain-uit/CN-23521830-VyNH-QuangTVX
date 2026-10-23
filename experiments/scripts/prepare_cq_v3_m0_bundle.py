"""Download and package the minimal CQ V3 train/dev evidence for M0.

The bundle deliberately excludes every test parquet, the 10.4M-row feature
shards, sequences, staging files, and the all-cohort target table.  It retains
only the course-disjoint split's train/dev model-ready and raw snapshots,
cutoff-audit tables, encoders, split metadata, and provenance documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


DATASET_HANDLE = "w4ngggg/phase-features-v3/versions/1"
REMOTE_ROOT = PurePosixPath("phase_features_v3/course_random_split")
PHASES = ("P1", "P2", "P3", "P4")

ROOT_FILES = (
    "BAO_CAO_PHASE_FEATURES_V3.md",
    "_SPLIT_SUCCESS.json",
    "_SUCCESS.json",
    "checkpoints.json",
    "feature_columns.json",
    "feature_dictionary.json",
    "load_data.py",
    "preprocessing.json",
    "run_manifest.json",
    "summary.json",
    "validation.json",
)
ENCODER_FILES = (
    "encoders/field_set.parquet",
    "encoders/learner.parquet",
    "encoders/school_set.parquet",
    "encoders/teacher_set.parquet",
)
TABLE_FILES = (
    "tables/course_splits.parquet",
    "tables/course_static_metadata.parquet",
)


def selected_relative_paths() -> list[str]:
    selected = [*ROOT_FILES, *ENCODER_FILES, *TABLE_FILES]
    for phase in PHASES:
        selected.extend(
            [
                f"audit/{phase}/part-000.parquet",
                f"model_ready/{phase}/train.parquet",
                f"model_ready/{phase}/dev.parquet",
                f"supervised_raw/{phase}/train.parquet",
                f"supervised_raw/{phase}/dev.parquet",
            ]
        )
    return sorted(selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inventory(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["relativeUrl"]: item for item in payload["files"]}


def write_readme(path: Path, *, remote_bytes: int, file_count: int) -> None:
    text = f"""# CQ V3 M0 development bundle

Generated from Kaggle dataset `{DATASET_HANDLE}`.

Purpose: M0 train/dev runs and mandatory development probes only.

Included:
- course-disjoint `course_random_split` metadata and encoders;
- P1--P4 train/dev `model_ready` snapshots;
- P1--P4 train/dev `supervised_raw` snapshots;
- P1--P4 cutoff audit tables;
- course split and static course metadata.

Intentionally excluded:
- every `test.parquet` file (test remains locked);
- 10.4M-row `features/` shards;
- `sequences/`, `_stage/`, and all-cohort target tables;
- the overlapping default user-course split.

Remote selected size: {remote_bytes} bytes across {file_count} source files.

Protocol warning: this bundle preserves the V3 random course-disjoint split.
It does not turn that split into the temporal or SGKF protocol requested by the
review.  Use for M0 development/conditional freeze only until the split policy
and `course_id == course_root_id` assumption are approved in writing.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.vendor_dir:
        sys.path.insert(0, str(args.vendor_dir.resolve()))
    import kagglehub

    inventory = read_inventory(args.inventory)
    relative_paths = selected_relative_paths()
    missing_remote = [
        str(REMOTE_ROOT / relative)
        for relative in relative_paths
        if str(REMOTE_ROOT / relative) not in inventory
    ]
    if missing_remote:
        raise FileNotFoundError(
            "Selected files missing from remote inventory:\n" + "\n".join(missing_remote)
        )

    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "source_download"
    bundle_dir = output_dir / "cq_v3_m0_dev"
    archive_path = output_dir / "cq_v3_m0_dev.zip"
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists() and args.force:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: list[dict[str, Any]] = []
    for relative in relative_paths:
        remote = str(REMOTE_ROOT / relative)
        target = bundle_dir / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or args.force:
            downloaded = Path(
                kagglehub.dataset_download(
                    DATASET_HANDLE,
                    path=remote,
                    output_dir=str(source_dir),
                    force_download=args.force,
                )
            )
            shutil.copy2(downloaded, target)

        expected_bytes = int(inventory[remote].get("totalBytes", 0))
        actual_bytes = target.stat().st_size
        if expected_bytes and actual_bytes != expected_bytes:
            raise ValueError(
                f"Size mismatch for {relative}: expected={expected_bytes}, actual={actual_bytes}"
            )
        manifest_files.append(
            {
                "path": relative.replace("\\", "/"),
                "bytes": actual_bytes,
                "sha256": sha256(target),
                "source_relative_url": remote,
            }
        )

    remote_bytes = sum(item["bytes"] for item in manifest_files)
    write_readme(
        bundle_dir / "README_M0_BUNDLE.md",
        remote_bytes=remote_bytes,
        file_count=len(manifest_files),
    )
    manifest = {
        "schema_version": "cq-v3-m0-dev-bundle-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_handle": DATASET_HANDLE,
        "remote_root": str(REMOTE_ROOT),
        "split_policy": "course_random_split; zero course_id overlap; not temporal or SGKF",
        "test_locked": True,
        "test_files_included": False,
        "source_file_count": len(manifest_files),
        "source_total_bytes": remote_bytes,
        "files": manifest_files,
    }
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(bundle_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(output_dir))

    result = {
        "bundle_dir": str(bundle_dir),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "source_file_count": len(manifest_files),
        "source_total_bytes": remote_bytes,
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
