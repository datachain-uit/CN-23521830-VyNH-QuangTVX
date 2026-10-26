"""Verify imported source bytes against the original pinned Git snapshot.

Read-only; does not fetch credentials, open research datasets, or edit code.
Additional organization documents are outside the imported-file manifest.
Use --index to verify staged/committed Git blob bytes rather than worktree files.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / "SOURCE_SNAPSHOT_SHA256SUMS.txt"
    seen: set[str] = set()
    mismatches: list[str] = []
    import json
    extension_record = root / "IMPLEMENTATION_CHANGES.json"
    extension = json.loads(extension_record.read_text(encoding="utf-8")) if extension_record.is_file() else {}
    overrides = extension.get("review1026_imported_overrides", {})
    permitted = {"experiments/pyproject.toml", "experiments/scripts/run_cq_v1_resource_calibration.py"}
    if not set(overrides) <= permitted:
        raise ValueError("Unregistered imported-source override")
    modified_imports = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or relative in seen:
            raise ValueError(f"Unsafe or duplicate manifest path: {relative}")
        seen.add(relative)
        if args.index:
            result = subprocess.run(
                ["git", "show", f":{relative}"], cwd=root,
                capture_output=True, check=False,
            )
            content = result.stdout if result.returncode == 0 else None
        else:
            content = target.read_bytes() if target.is_file() else None
        actual = hashlib.sha256(content).hexdigest() if content is not None else None
        declared = overrides.get(relative)
        if declared:
            if declared.get("original_sha256") != expected or actual != declared.get("current_sha256"):
                mismatches.append(relative)
            modified_imports += 1
        elif actual != expected:
            mismatches.append(relative)
    if not seen:
        raise ValueError("Source manifest is empty")
    if mismatches:
        print("FAIL: imported source differs:")
        for relative in mismatches:
            print(relative)
        return 1
    extension_count = 0
    if extension_record.is_file():
        for relative, expected in extension["current_file_sha256"].items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or relative in seen:
                raise ValueError(f"Unsafe or overlapping extension path: {relative}")
            if args.index:
                result = subprocess.run(["git", "show", f":{relative}"], cwd=root, capture_output=True, check=False)
                content = result.stdout if result.returncode == 0 else None
            else:
                content = target.read_bytes() if target.is_file() else None
            if content is None or hashlib.sha256(content).hexdigest() != expected:
                mismatches.append(relative)
            extension_count += 1
        if mismatches:
            print("FAIL: extension bytes differ: " + ", ".join(mismatches))
            return 1
    surface = "Git index" if args.index else "worktree"
    if set(overrides) - seen:
        raise ValueError("Override does not refer to an imported file")
    print(f"PASS: {len(seen)-modified_imports} unchanged imported files, "
          f"{modified_imports} explicitly modified imports and {extension_count} declared extension files ({surface})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
