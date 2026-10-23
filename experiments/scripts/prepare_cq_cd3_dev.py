"""Prepare a fresh CD3 dev-only bundle and runtime without overwriting V4.

Requires data-team raw_features layout plus the original immutable V4 dev
bundle for spine comparison. No test payload is accepted or copied.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from skq_exp.artifacts import atomic_json, sha256_file, stable_hash
from skq_exp.data.adapters.course_quality_v4 import CourseQualityV4Adapter, _verify_sha256s
from skq_exp.data.cq_cd3_views import build_cd3_views, PHASES


def prepare(raw_root: Path, v4_bundle: Path, bundle: Path, processed_root: Path) -> Path:
    if bundle.exists() or (processed_root / "course_quality_v4").exists():
        raise FileExistsError("Use fresh CD3 output paths; do not overwrite a prior runtime")
    for source in (raw_root, v4_bundle):
        if any(p.suffix.lower() in {".parquet", ".csv", ".npy", ".npz"}
               and "test" in str(p.relative_to(source)).lower() for p in source.rglob("*")):
            raise ValueError(f"Test payload is not allowed in dev input: {source}")
        _verify_sha256s(source)
    raw = raw_root / "raw_features"
    paths = {"train": "pool_b/train.parquet", "fulltrain_a": "fulltrain_a/train.parquet",
             **{f"dev_phase{i}": f"pool_b/dev_phase{i}.parquet" for i in range(1, 5)}}
    frames = {name: pd.read_parquet(raw / path) for name, path in paths.items()}
    # Check against original V4 rows before fitting anything; never create a new split/cohort.
    identity = ["row_id", "enrollment_id", "user_id", "course_id", "course_family_id", "y_alpha_1"]
    for name, path in paths.items():
        original = pd.read_parquet(v4_bundle / path)
        a, b = frames[name], original
        if a.row_id.duplicated().any() or b.row_id.duplicated().any():
            raise ValueError(f"{name}: duplicate row IDs")
        for col in identity + ["q_alpha_1", "enroll_time", "cutoff_time"]:
            if col not in a or col not in b:
                raise ValueError(f"{name}: missing immutable field {col}")
            left = a.set_index("row_id", drop=False)[col].sort_index()
            right = b.set_index("row_id", drop=False)[col].sort_index()
            if not left.equals(right):
                raise ValueError(f"{name}: immutable V4 spine mismatch in {col}")
    ready, audit, contract, prep = build_cd3_views(
        frames["train"], {p: frames[f"dev_phase{p[-1]}"] for p in PHASES}, frames["fulltrain_a"],
    )
    bundle.mkdir(parents=True)
    for name in ("splits.json", "run_manifest.json", "family_splits.parquet", "course_splits.parquet"):
        shutil.copy2(v4_bundle / name, bundle / name)
    for name, path in paths.items():
        target = bundle / path
        target.parent.mkdir(parents=True, exist_ok=True)
        ready[name].to_parquet(target, index=False)
    audit.to_csv(bundle / "qa23_by_phase.csv", index=False)
    atomic_json(bundle / "held_out_columns.json", contract["held_out_columns"])
    atomic_json(bundle / "selector_feature_columns.json", contract["selector_columns"])
    atomic_json(bundle / "learner_feature_columns.json", contract["learner_columns"])
    atomic_json(bundle / "preprocessing.json", prep)
    atomic_json(bundle / "validation.json", {"dataset_version": "CQ V4", "test_excluded": True,
                                            "feature_view": "cd3", "immutable_spine_matches": True})
    atomic_json(bundle / "bundle_manifest.json", {"dataset_version": "CQ V4", "test_excluded": True,
        "feature_view": "cd3", "raw_source_sha256": sha256_file(raw_root / "SHA256SUMS.txt"),
        "raw_input_files_sha256": {path: sha256_file(raw / path) for path in paths.values()},
        "v4_source_sha256": sha256_file(v4_bundle / "SHA256SUMS.txt"), "contract": contract})
    hashes = [f"{sha256_file(p)}  {p.relative_to(bundle).as_posix()}" for p in sorted(bundle.rglob("*")) if p.is_file()]
    (bundle / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    result = CourseQualityV4Adapter().prepare(bundle, processed_root)
    runtime = result.parent
    adapter_contract = json.loads((runtime / "feature_contract.json").read_text(encoding="utf-8"))
    adapter_contract.update(contract)
    adapter_contract.pop("contract_hash", None)
    adapter_contract["contract_hash"] = stable_hash(adapter_contract)
    atomic_json(runtime / "feature_contract.json", adapter_contract)
    atomic_json(runtime / "schema.json", adapter_contract)
    manifest = json.loads(result.read_text(encoding="utf-8"))
    manifest.update(feature_view="cd3", feature_contract_hash=adapter_contract["contract_hash"],
                    raw_source_sha256=sha256_file(raw_root / "SHA256SUMS.txt"))
    atomic_json(result, manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--v4-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.raw_root.resolve(), args.v4_bundle.resolve(), args.output_bundle.resolve(), args.processed_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
