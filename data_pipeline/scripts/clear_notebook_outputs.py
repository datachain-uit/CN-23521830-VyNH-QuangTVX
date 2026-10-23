"""Clear outputs from publishable root notebooks; keep ignored local backups."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cleaned = []
    for path in sorted(ROOT.glob("*.ipynb")):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path.name], cwd=ROOT
        )
        if ignored.returncode == 0:
            continue
        if ignored.returncode != 1:
            raise RuntimeError("Cannot determine notebook ignore status")
        original = path.read_bytes()
        notebook = json.loads(original)
        sources = [(c.get("cell_type"), c.get("source")) for c in notebook["cells"]]
        changed = False
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    changed = True
                cell["outputs"] = []
                cell["execution_count"] = None
            metadata = cell.get("metadata", {})
            for key in ("execution", "ExecuteTime", "trusted", "collapsed", "scrolled"):
                if key in metadata:
                    metadata.pop(key)
                    changed = True
        if "widgets" in notebook.get("metadata", {}):
            notebook["metadata"].pop("widgets")
            changed = True
        if not changed:
            continue
        assert sources == [(c.get("cell_type"), c.get("source")) for c in notebook["cells"]]
        checksum = hashlib.sha256(original).hexdigest()
        backup = ROOT / ".local_notebook_backups" / checksum / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        assert backup.read_bytes() == original
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        cleaned.append(path.name)
    print(json.dumps({"cleaned": cleaned, "backups": ".local_notebook_backups/"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
