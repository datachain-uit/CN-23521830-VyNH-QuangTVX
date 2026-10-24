"""Check Git candidates or staged blobs without printing private file contents."""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 2 * 1024 * 1024
PRIVATE_DIRS = {
    "input", "data", "datasets", "artifacts", "_stage", "_scratch", "encoders",
    "pretrained", "models", "checkpoints", "source_snapshot", ".local_notebook_backups",
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".cache", ".venv", "venv", ".aws", ".ssh",
}
PRIVATE_PREFIXES = ("final_quality", "phase_features", "weighted_loqv", "fullA", "fulltrain_a", "cq_v4")
SECRET_PATTERNS = {
    "private-key marker": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access-key pattern": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token pattern": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "API-key pattern": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b"),
    "literal learner ID": re.compile(r"\bU_\d+\b"),
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def path_problem(name):
    path = PurePosixPath(name)
    if name == "afelo_sentiment_cardiff/requirements-sentiment.txt":
        return None
    if name == "17_eda_final_quality_v2.ipynb":
        return "private example notebook; publish its Python implementation instead"
    if any(p in PRIVATE_DIRS or p.startswith(PRIVATE_PREFIXES) for p in path.parts[:-1]):
        return "dataset/cache/output directory"
    if path.name in {".gitignore", "pyproject.toml"}:
        return None
    if name in {".gitattributes", "README.md", "GITHUB_CODE_ONLY.md", "HANDOFF.md"}:
        return None
    if fnmatch.fnmatchcase(path.name, "requirements*.txt"):
        return None
    if path.suffix in {".py", ".ipynb", ".sh"}:
        return None
    return "outside the code-only file allowlist"


def content_problems(name, payload):
    errors = []
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError:
        return ["not UTF-8 source text"]
    if "\0" in source:
        return ["binary content in a source file"]
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(source):
            errors.append(label)
    if name.endswith(".ipynb"):
        try:
            notebook = json.loads(source)
            cells = notebook["cells"]
            if any(c.get("outputs") or c.get("execution_count") is not None for c in cells if c.get("cell_type") == "code"):
                errors.append("notebook has outputs/execution counts; run scripts/clear_notebook_outputs.py")
            if any(c.get("attachments") for c in cells):
                errors.append("notebook contains embedded attachments")
            if notebook.get("metadata", {}).get("widgets"):
                errors.append("notebook contains widget state")
        except (ValueError, KeyError, TypeError):
            errors.append("invalid notebook JSON/schema")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--working-tree", action="store_true", help="Check files Git would allow, before staging")
    mode.add_argument("--staged", action="store_true", help="Check actual index blobs, including force-added files (default)")
    args = parser.parse_args()
    if args.working_tree:
        names = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""})
        names = [n for n in names if (ROOT / n).exists() or (ROOT / n).is_symlink()]
    else:
        names = sorted(set(git("ls-files", "--cached", "-z").decode().split("\0")) - {""})
    errors = []
    for name in names:
        reason = path_problem(name)
        if reason:
            errors.append((name, reason))
            continue
        if args.working_tree:
            path = ROOT / name
            if path.is_symlink():
                errors.append((name, "symlink; target not checked"))
                continue
            size = path.stat().st_size
            read = path.read_bytes
        else:
            stage = git("ls-files", "--stage", "--", name).decode()
            if not stage.startswith("100644 ") and not stage.startswith("100755 "):
                errors.append((name, "non-regular or unmerged index entry"))
                continue
            # Resolve index paths from this pipeline directory even when it is
            # nested inside a larger Git repository (for example data_pipeline/).
            index_path = ":./" + name
            size = int(git("cat-file", "-s", index_path))
            read = lambda: git("show", index_path)
        if size > MAX_SOURCE_BYTES:
            errors.append((name, "source exceeds 2 MiB; review separately"))
            continue
        errors.extend((name, reason) for reason in content_problems(name, read()))
    for name, reason in errors:
        print(f"BLOCKED {name}: {reason}")
    if errors:
        print(f"FAILED: {len(errors)} issue(s), {len(names)} file(s) checked. No file contents printed.")
        return 1
    if not names:
        print("No staged files. Use --working-tree to inspect candidates before git add.")
    else:
        print(f"PASS: {len(names)} code/configuration files checked ({'working tree' if args.working_tree else 'staged blobs'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
