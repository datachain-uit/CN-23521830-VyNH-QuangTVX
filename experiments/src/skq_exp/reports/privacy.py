"""Audit gói public mà không đọc hoặc xuất nội dung dữ liệu nghiên cứu.

Audit kiểm ba bề mặt độc lập: file Git đang theo dõi, tên object từng xuất hiện
trong lịch sử Git và các entry của archive chuẩn bị công bố. Mỗi finding giữ
nguyên nguồn/phân loại; không tự xóa file, rewrite history hay gộp lỗi.
"""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import atomic_json


SCANNER_VERSION = "1.0.0"
DENIED_SUFFIXES = {
    ".npy", ".npz", ".parquet", ".joblib", ".pkl", ".pickle",
    ".tar", ".gz", ".zip", ".xlsx", ".xls", ".docx",
}
DENIED_PARTS = {"results", "artifacts_from_kaggle", "catboost_info"}
DENIED_BASENAME_MARKERS = {
    "predictions", "probabilities", "selected_indices", "sample_weights",
    "row_ids", "student_ids", "learner_ids_private",
}
ALLOWLIST = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "artifacts/README.md",
    "external/repos/.gitkeep",
}
SECRET_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "kaggle_key": re.compile(r"(?i)KAGGLE_KEY\s*[=:]\s*['\"]?[A-Za-z0-9_-]{16,}"),
}


def audit_public_release(
    project_root: str | Path,
    *,
    archives: Iterable[str | Path] = (),
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Chạy audit read-only và tùy chọn ghi evidence JSON."""
    root = Path(project_root).resolve()
    tracked = _git_lines(root, ["git", "ls-files"])
    history_objects = _git_lines(root, ["git", "rev-list", "--objects", "--all"])
    commit = _git_scalar(root, ["git", "rev-parse", "HEAD"])

    tracked_findings = []
    for relative in tracked:
        finding = classify_public_path(relative)
        if finding:
            tracked_findings.append({"path": relative, **finding})
        tracked_findings.extend(_content_findings(root, relative))

    history_findings = []
    for row in history_objects:
        parts = row.split(" ", 1)
        if len(parts) != 2:
            continue
        finding = classify_public_path(parts[1])
        if finding:
            history_findings.append({"object": parts[0], "path": parts[1], **finding})

    archive_findings = []
    for archive_path in archives:
        archive_findings.extend(_audit_archive(Path(archive_path)))

    findings = tracked_findings + history_findings + archive_findings
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "scanner_version": SCANNER_VERSION,
        "code_commit": commit,
        "status": "pass" if not findings else "fail",
        "tracked_findings": tracked_findings,
        "history_findings": history_findings,
        "archive_findings": archive_findings,
        "allowlist": sorted(ALLOWLIST),
        "denylist": {
            "suffixes": sorted(DENIED_SUFFIXES),
            "path_parts": sorted(DENIED_PARTS),
            "basename_markers": sorted(DENIED_BASENAME_MARKERS),
            "secret_patterns": sorted(SECRET_PATTERNS),
        },
        "resolution": (
            "Không phát hiện artifact public bị cấm."
            if not findings
            else "Phải phân loại và xử lý từng finding; không tự rewrite lịch sử Git."
        ),
        "reviewer": None,
    }
    if output is not None:
        atomic_json(output, evidence)
    return evidence


def classify_public_path(path: str) -> dict[str, str] | None:
    """Phân loại tên path; không mở file và không suy success từ ngoại lệ."""
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized in ALLOWLIST:
        return None
    candidate = Path(normalized)
    lowered_parts = {part.lower() for part in candidate.parts}
    lowered_name = candidate.name.lower()
    if normalized.startswith(("data/raw/", "data/processed/")):
        return {"kind": "private_data_path", "reason": "raw/processed data không public"}
    if lowered_parts & DENIED_PARTS:
        return {"kind": "private_artifact_path", "reason": "thư mục kết quả private"}
    if any(lowered_name.endswith(suffix) for suffix in DENIED_SUFFIXES):
        return {"kind": "private_artifact_type", "reason": "kiểu file không thuộc public allowlist"}
    if any(marker in lowered_name for marker in DENIED_BASENAME_MARKERS):
        return {"kind": "row_level_artifact", "reason": "tên file gợi ý artifact mức dòng"}
    if candidate.suffix.lower() == ".ipynb":
        return {"kind": "notebook_requires_output_check", "reason": "notebook phải sạch output"}
    return None


def _content_findings(root: Path, relative: str) -> list[dict[str, str]]:
    path = root / relative
    if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        return []
    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return [{"path": relative, "kind": "notebook_unreadable", "reason": "không kiểm được output"}]
        if any(cell.get("outputs") for cell in notebook.get("cells", [])):
            return [{"path": relative, "kind": "notebook_output", "reason": "notebook còn output"}]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings = []
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            findings.append({"path": relative, "kind": name, "reason": "khớp mẫu credential"})
    return findings


def _audit_archive(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return [{"archive": str(path), "kind": "archive_missing", "reason": "không tìm thấy archive"}]
    findings = []
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                finding = classify_public_path(name.removeprefix("project/"))
                if finding:
                    findings.append({"archive": str(path), "path": name, **finding})
    except (OSError, zipfile.BadZipFile) as error:
        findings.append({"archive": str(path), "kind": "archive_unreadable", "reason": repr(error)})
    return findings


def _git_lines(root: Path, command: list[str]) -> list[str]:
    completed = subprocess.run(
        command, cwd=root, check=True, text=True, capture_output=True, timeout=60,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _git_scalar(root: Path, command: list[str]) -> str:
    values = _git_lines(root, command)
    return values[0] if values else "unknown"
