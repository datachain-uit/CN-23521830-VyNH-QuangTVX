"""Bổ sung theo Review-2026-10-31(dùng độc lập, chỉ cần thư viện chuẩn của Python).

1. find_test_files, build_seal_manifest, verify_seal: niêm phong các tệp test trước M1 (QĐ42).
   find_test_files đi theo symlink, vì bản local của nhóm giữ một số thư mục dữ liệu
   dưới dạng symlink (Path.rglob không đi vào thư mục symlink).
2. parse_sha256sums, compare_hash_sets, compare_protected: đối chiếu hash của gói dữ liệu với
   đầu vào của panel (QĐ39e, Y45). compare_protected chỉ so các artifact được bảo vệ, vì tài
   liệu và manifest đổi khi cập nhật data card.
3. payload_guard: chặn tệp dữ liệu, con trỏ Git LFS và luật LFS trong .gitattributes trước
   khi commit vào repo công khai (QĐ40). Bộ kiểm tra trong PR #5 cho PASS 298 con trỏ LFS vì
   chúng có trong PUBLICATION_MANIFEST.json do nhóm tự lập; checker của PR #4 chặn theo đường
   dẫn nhưng không kiểm .gitattributes. Hàm này coi chính con trỏ LFS và luật filter=lfs là
   lỗi, bất kể manifest.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# 1. Niêm phong tệp test (QĐ42) ---------------------------------------------------------------
TEST_PATTERNS = (
    "*test_phase[1-4].parquet",     # V4.2 raw_features/model_ready
    "*/test.parquet", "test.parquet",  # V4: model_ready|sequences|supervised_raw/P1–P4/test.parquet
    "*/audit/test/*",               # audit của phần test
    "*/test/*.parquet",
    "*_test.parquet", "*test_*.parquet",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_test_files(root, patterns=TEST_PATTERNS) -> list:
    """Đường dẫn tương đối (posix, đã sắp xếp) của mọi tệp khớp một mẫu test dưới root.
    Đi theo symlink, mỗi thư mục thật chỉ duyệt một lần (tránh vòng lặp). Mẫu thiên về bắt
    thừa (ví dụ “latest_x.parquet” cũng khớp “*test_*”); sinh viên thực hiện niêm phong duyệt lại danh sách
    trước khi gọi build_seal_manifest."""
    root = Path(root)
    out, seen = [], set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat) for pat in patterns):
                    out.append(rel)
    return sorted(out)


def _canonical(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def build_seal_manifest(root, files, sealed_by: str, note: str = "") -> dict:
    """SHA256 và kích thước của từng tệp; manifest_sha256 là hash của toàn bộ danh sách.
    Ghi manifest_sha256 vào hồ sơ M1 để không ai sửa tệp test mà không bị phát hiện."""
    if not sealed_by.strip():
        raise ValueError("cần họ tên và MSSV của sinh viên thực hiện niêm phong")
    root = Path(root)
    entries = {rel: {"sha256": sha256_file(root / rel), "bytes": (root / rel).stat().st_size}
               for rel in sorted(files)}
    return {"schema": "cq-test-seal-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "sealed_by": sealed_by, "note": note, "n_files": len(entries), "files": entries,
            "manifest_sha256": _canonical(entries)}


def verify_seal(root, manifest: dict) -> dict:
    """Kiểm lại sau freeze: mọi tệp còn đủ, đúng hash, và manifest chưa bị sửa."""
    root = Path(root)
    missing, changed = [], []
    for rel, meta in manifest["files"].items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256_file(p) != meta["sha256"]:
            changed.append(rel)
    intact = _canonical(manifest["files"]) == manifest["manifest_sha256"]
    return {"ok": intact and not missing and not changed, "manifest_intact": intact,
            "missing": missing, "changed": changed}


# 2. Đối chiếu hash (QĐ39e) -------------------------------------------------------------------
def parse_sha256sums(text: str) -> dict:
    """Đọc định dạng `sha256sum`: '<hash>  <đường dẫn>' (bỏ dấu * của chế độ nhị phân)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, path = line.partition(" ")
        out[path.strip().lstrip("*").replace("\\", "/")] = digest.lower()
    return out


def compare_hash_sets(a: dict, b: dict, key=lambda p: p) -> dict:
    """So hai tập {đường dẫn: hash}. key cho phép so theo tên chuẩn hóa (ví dụ bỏ thư mục gốc)."""
    ka = {key(p): h for p, h in a.items()}
    kb = {key(p): h for p, h in b.items()}
    common = set(ka) & set(kb)
    mismatched = sorted(k for k in common if ka[k] != kb[k])
    return {"matched": len(common) - len(mismatched), "mismatched": mismatched,
            "only_a": sorted(set(ka) - set(kb)), "only_b": sorted(set(kb) - set(ka)),
            "identical": not mismatched and set(ka) == set(kb)}


def compare_protected(protected: dict, other: dict, key=lambda p: p) -> dict:
    """Chỉ so các artifact được bảo vệ (ví dụ model_input_sha256 trong data_card_evidence.json)
    với một tập hash khác; tệp ngoài danh sách bảo vệ (tài liệu, manifest) bị bỏ qua."""
    kp = {key(p): h.lower() for p, h in protected.items()}
    ko = {key(p): h.lower() for p, h in other.items()}
    common = sorted(set(kp) & set(ko))
    return {"matched": [k for k in common if kp[k] == ko[k]],
            "mismatched": [k for k in common if kp[k] != ko[k]],
            "missing_in_other": sorted(set(kp) - set(ko))}


# 3. Chặn dữ liệu trước khi commit (QĐ40) --------------------------------------------------------
DATA_SUFFIXES = (".parquet", ".csv", ".tsv", ".npz", ".npy", ".feather", ".arrow", ".pkl",
                 ".pickle", ".joblib", ".h5", ".zip", ".tar", ".gz", ".7z", ".xlsx", ".db",
                 ".jsonl", ".ndjson", ".pt", ".pth", ".safetensors", ".duckdb", ".sqlite",
                 ".sqlite3", ".hdf5", ".xls", ".tgz", ".bz2", ".xz")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def payload_guard(files: dict, gitattributes_text: str = "") -> dict:
    """files: {đường dẫn: nội dung bytes trong index}. Lỗi khi có tệp đuôi dữ liệu, con trỏ
    Git LFS (dù là văn bản) hoặc luật filter=lfs trong .gitattributes."""
    problems = []
    for path, payload in sorted(files.items()):
        low = path.lower()
        if low.endswith(DATA_SUFFIXES):
            problems.append((path, "tệp dữ liệu"))
        elif payload[:len(LFS_POINTER_PREFIX)] == LFS_POINTER_PREFIX:
            problems.append((path, "con trỏ Git LFS"))
    for n, line in enumerate(gitattributes_text.splitlines(), 1):
        if not line.lstrip().startswith("#") and "filter=lfs" in line.split()[1:]:
            problems.append((f".gitattributes:{n}", "luật Git LFS"))
    return {"pass": not problems, "problems": problems}
