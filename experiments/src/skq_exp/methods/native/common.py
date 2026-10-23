"""File này cung cấp thao tác chung cho adapter repo ngoài.

Nó kiểm tra vị trí repo, lấy commit Git và quản lý sys.path tạm thời. Mục đích là
giữ provenance rõ mà không cài code repo ngoài vào package SKQ.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def verify_locked_repo(repo: Path) -> str:
    """Trả commit khi checkout đúng lock; sai/mất lock thì dừng adapter native."""
    repo = Path(repo).resolve()
    lock_path = repo.parent.parent / "official_repos.lock.json"
    if not lock_path.exists():
        raise RuntimeError(f"Thiếu official repo lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = lock.get(repo.name)
    if not isinstance(entry, dict) or not entry.get("commit"):
        raise RuntimeError(f"Repo {repo.name} chưa có commit đầy đủ trong lock")
    actual = git_commit(repo)
    expected = str(entry["commit"])
    if actual != expected:
        raise RuntimeError(f"Repo {repo.name} ở commit {actual}, cần đúng {expected}")
    return actual


@contextlib.contextmanager
def prepend_sys_path(path: Path):
    old = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old
