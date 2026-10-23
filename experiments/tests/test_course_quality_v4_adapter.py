"""Kiểm tra adapter CQ V4 nhận đúng bundle dev-only do nhóm dữ liệu bàn giao."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skq_exp.data.adapters import prepare_external_dataset
from skq_exp.data.prepare import load_processed


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_hashes(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _frame(rows: list[tuple[str, str, str, str]], *, split: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["user_id", "course_id", "course_family_id", "phase"])
    frame["split"] = split
    frame["row_id"] = [f"{split}-{user}-{phase}" for user, _, _, phase in rows]
    frame["enrollment_id"] = [f"{user}-{course}" for user, course, _, _ in rows]
    frame["y_alpha_1"] = (["E", "A"] * (len(frame) // 2 + 1))[:len(frame)]
    frame["q_alpha_1"] = np.where(frame["y_alpha_1"] == "E", 0.1, 0.6)
    frame["f0"] = np.arange(len(frame), dtype=float)
    frame["phase_fraction"] = frame["phase"].map(
        {"P1": 0.25, "P2": 0.50, "P3": 0.75, "P4": 0.90, "lambda1": 1.0}
    )
    frame["learner_code"] = 1.0
    return frame


def test_course_quality_v4_runtime_bundle_contract(tmp_path: Path) -> None:
    source = tmp_path / "runtime_dev_only"
    processed = tmp_path / "processed"
    (source / "pool_b").mkdir(parents=True)
    (source / "fulltrain_a").mkdir()

    _write_json(source / "validation.json", {"dataset_version": "CQ V4", "checks_passed": 1})
    _write_json(source / "bundle_manifest.json", {
        "dataset_version": "CQ V4", "scope": "train/dev only", "test_excluded": True,
    })
    _write_json(source / "splits.json", {
        "dataset_version": "CQ V4",
        "seed": 42,
        "assignment": {"fam-train": "train", "fam-dev": "dev"},
        "course_assignment": {
            "c-train": {"course_family_id": "fam-train", "split": "train"},
            "c-dev": {"course_family_id": "fam-dev", "split": "dev"},
        },
    })
    for name in ("run_manifest.json", "preprocessing.json"):
        _write_json(source / name, {})
    _write_json(source / "selector_feature_columns.json", {"columns": ["f0"]})
    _write_json(
        source / "learner_feature_columns.json",
        {"columns": ["f0", "phase_fraction", "learner_code"]},
    )
    pd.DataFrame({"course_family_id": ["fam-train", "fam-dev"]}).to_parquet(
        source / "family_splits.parquet", index=False,
    )
    pd.DataFrame({
        "course_id": ["c-train", "c-dev"],
        "course_family_id": ["fam-train", "fam-dev"],
    }).to_parquet(source / "course_splits.parquet", index=False)

    train_rows = [
        (user, "c-train", "fam-train", phase)
        for phase in ("P1", "P2", "P3", "P4")
        for user in ("u1", "u2")
    ]
    _frame(train_rows, split="train").to_parquet(source / "pool_b" / "train.parquet", index=False)
    for position, phase in enumerate(("P1", "P2", "P3", "P4"), start=1):
        _frame([("u3", "c-dev", "fam-dev", phase)], split="dev").to_parquet(
            source / "pool_b" / f"dev_phase{position}.parquet", index=False,
        )
    _frame([
        ("u1", "c-train", "fam-train", "lambda1"),
        ("u2", "c-train", "fam-train", "lambda1"),
    ], split="train").to_parquet(source / "fulltrain_a" / "train.parquet", index=False)
    _write_hashes(source)

    manifest = prepare_external_dataset("course_quality_v4", source, processed)
    data = load_processed("course_quality_v4", processed, mmap=False)

    assert data["X_train"].shape == (8, 3)
    assert data["X_selector_train"].shape == (8, 1)
    assert all(data[f"X_dev_phase{phase}"].shape == (1, 3) for phase in range(1, 5))
    assert data["X_fulltrain_a_train"].shape == (2, 3)
    assert "X_test" not in data
    assert "course_groups_dev_phase1" in data
    assert "course_family_groups_dev_phase1" in data
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "ready_for_m0_v4_dev"
    assert payload["dataset_version"] == "CQ V4"
