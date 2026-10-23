"""Script này clone đúng repo/commit trong lock file vào external/repos.

Chạy một lần sau khi đưa project lên GitHub/Kaggle. Repo đã đúng commit được giữ
nguyên; thư mục sai remote hoặc sai commit sẽ dừng để người dùng kiểm tra thủ công.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LOCK = json.loads((PROJECT / "external" / "official_repos.lock.json").read_text(encoding="utf-8"))
ROOT = PROJECT / "external" / "repos"


def _git(target: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(target), *args], text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _normalized_remote(value: str) -> str:
    return value.rstrip("/").removesuffix(".git").lower()


def _validate_checkout_identity(name: str, target: Path, expected_url: str) -> None:
    try:
        top = Path(_git(target, "rev-parse", "--show-toplevel")).resolve()
        remote = _git(target, "remote", "get-url", "origin")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{name}: checkout dở ở {target}; xóa riêng thư mục này rồi chạy lại"
        ) from error
    if top != target.resolve():
        raise RuntimeError(
            f"{name}: {target} không phải Git top-level (đang trỏ tới {top}); "
            "không được dùng repo cha làm source"
        )
    if _normalized_remote(remote) != _normalized_remote(expected_url):
        raise RuntimeError(
            f"{name}: origin {remote!r} không khớp official URL {expected_url!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", action="append", choices=tuple(LOCK),
        help="Chỉ fetch repo được nêu; có thể lặp. Mặc định fetch toàn lock.",
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.repo or LOCK.keys())
    for name in selected:
        entry = LOCK[name]
        target = ROOT / name
        if not target.exists():
            clone = ["git", "clone"]
            if entry.get("sparse_paths"):
                clone.extend(["--filter=blob:none", "--sparse"])
            subprocess.run([*clone, entry["url"], str(target)], check=True)
        _validate_checkout_identity(name, target, entry["url"])
        if entry.get("sparse_paths"):
            subprocess.run(
                ["git", "-C", str(target), "sparse-checkout", "set", *entry["sparse_paths"]],
                check=True,
            )
        try:
            commit = _git(target, "rev-parse", "HEAD")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"{name}: checkout dở ở {target}; xóa riêng thư mục này rồi chạy lại"
            ) from error
        expected = str(entry["commit"])
        if commit != expected:
            subprocess.run(
                ["git", "-C", str(target), "fetch", "origin", expected], check=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", expected], check=True,
            )
            commit = _git(target, "rev-parse", "HEAD")
        if commit != expected:
            raise RuntimeError(f"{name}: commit {commit} không khớp full lock {expected}")
        print(f"OK {name}: {commit}")


if __name__ == "__main__":
    main()
