"""Kiểm thử cq_patch_1031.py(Review-2026-10-31). Chạy: pytest -q test_cq_patch_1031.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # data_pipeline/ khi đặt trong tests/

from cq_patch_1031 import (  # noqa: E402
    find_test_files, build_seal_manifest, verify_seal, parse_sha256sums, compare_hash_sets,
    compare_protected, payload_guard)


def _bundle(tmp_path):
    """Gói giả lập: V4.2 là thư mục thật; phase_features_v4 là symlink như bản local của nhóm."""
    root = tmp_path / "data_pipeline"
    files = {
        "raw_features/pool_b/train_phase1.parquet": b"train",
        "raw_features/pool_b/dev_phase1.parquet": b"dev",
        "raw_features/pool_b/test_phase1.parquet": b"test-1",
        "model_ready/pool_b/test_phase4.parquet": b"test-4",
        "audit/test/validation.json": b"{}",
        "audit/qa23_by_phase.json": b"{}",
        "final_quality_v4/latest.parquet": b"khong-phai-test",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    store = tmp_path / "private_store" / "phase_features_v4" / "model_ready" / "P1"
    store.mkdir(parents=True)
    (store / "test.parquet").write_bytes(b"v4-test")
    (store / "train.parquet").write_bytes(b"v4-train")
    (root / "phase_features_v4").symlink_to(store.parents[1], target_is_directory=True)
    (root / "vong_lap").symlink_to(root, target_is_directory=True)   # symlink vòng lặp
    return root


def test_find_test_files_only_test(tmp_path):
    root = _bundle(tmp_path)
    assert find_test_files(root) == ["audit/test/validation.json",
                                     "model_ready/pool_b/test_phase4.parquet",
                                     "phase_features_v4/model_ready/P1/test.parquet",
                                     "raw_features/pool_b/test_phase1.parquet"]


def test_seal_detects_change_and_missing(tmp_path):
    root = _bundle(tmp_path)
    m = build_seal_manifest(root, find_test_files(root), sealed_by="Họ tên, MSSV")
    assert m["n_files"] == 4 and verify_seal(root, m)["ok"]
    (root / "raw_features/pool_b/test_phase1.parquet").write_bytes(b"sua")
    r = verify_seal(root, m)
    assert not r["ok"] and r["changed"] == ["raw_features/pool_b/test_phase1.parquet"]
    (root / "audit/test/validation.json").unlink()
    assert verify_seal(root, m)["missing"] == ["audit/test/validation.json"]
    m["files"]["model_ready/pool_b/test_phase4.parquet"]["sha256"] = "0" * 64
    assert not verify_seal(root, m)["manifest_intact"]
    with pytest.raises(ValueError):
        build_seal_manifest(root, [], sealed_by=" ")


def test_compare_hash_sets_with_sha256sums():
    a = parse_sha256sums("aa  cd3/pool_b/train.parquet\nbb *cd3/pool_b/dev_phase1.parquet\n")
    b = parse_sha256sums("aa  v4_2/pool_b/train.parquet\ncc  v4_2/pool_b/dev_phase1.parquet\n")
    strip = lambda p: p.split("/", 1)[1]                     # bỏ thư mục gốc khác nhau
    r = compare_hash_sets(a, b, key=strip)
    assert r["matched"] == 1 and r["mismatched"] == ["pool_b/dev_phase1.parquet"]
    assert not r["identical"]
    assert compare_hash_sets(a, a)["identical"]


def test_compare_protected_ignores_docs_and_reports_missing_test():
    protected = {"raw_features/pool_b/train.parquet": "AA", "preprocessing.json": "bb",
                 "raw_features/pool_b/test_phase1.parquet": "cc"}
    panel = parse_sha256sums("aa  cd3/raw_features/pool_b/train.parquet\n"
                             "bb  cd3/preprocessing.json\n"
                             "ff  cd3/README.md\n")          # tài liệu đổi hash: không so
    key = lambda p: p.removeprefix("cd3/")
    r = compare_protected(protected, panel, key=key)
    assert r["matched"] == ["preprocessing.json", "raw_features/pool_b/train.parquet"]
    assert r["mismatched"] == [] and r["missing_in_other"] == ["raw_features/pool_b/test_phase1.parquet"]
    panel["cd3/preprocessing.json"] = "00"
    assert compare_protected(protected, panel, key=key)["mismatched"] == ["preprocessing.json"]


def test_payload_guard_blocks_lfs_pointers_and_data():
    pointer = (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 +
               b"\nsize 133\n")
    r = payload_guard({"data_pipeline/x/part-00.parquet": pointer,
                       "data_pipeline/notes/ptr.bin": pointer,
                       "data_pipeline/25_build_phase_features_v4_2.py": b"print(1)\n"},
                      "data_pipeline/**/*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    kinds = {k for _, k in r["problems"]}
    assert not r["pass"] and kinds == {"tệp dữ liệu", "con trỏ Git LFS", "luật Git LFS"}
    ok = payload_guard({"data_pipeline/README.md": b"# code only\n"}, "*.py text\n")
    assert ok["pass"]
