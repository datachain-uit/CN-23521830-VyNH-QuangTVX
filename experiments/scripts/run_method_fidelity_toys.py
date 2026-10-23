"""Chạy toy reproduction G10 cho toàn bộ baseline bảng chính.

Artifact chỉ giữ trạng thái, provenance, budget và diagnostics; không ghi row ID
hay tập con đã chọn. Một dependency/source lỗi được giữ nguyên là blocked/failed,
không fallback sang proxy. Lệnh strict audit chỉ pass khi cả mười dòng success.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from skq_exp.methods.fidelity import MAIN_BASELINE_IDS, audit_fidelity_manifest
from skq_exp.methods.registry import build_selector


PROJECT = Path(__file__).resolve().parents[1]


def _toy(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        weights=(0.50, 0.30, 0.20),
        class_sep=1.25,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # Bốn phase giả chỉ để kiểm contract C03; không phải evidence CQ.
    phases = np.asarray([f"P{index % 4 + 1}" for index in range(len(y))])
    return X, y.astype(np.int64), phases


def _options(method_id: str, device: str) -> dict[str, Any]:
    if method_id == "b_ccs_oof_xgb_adapted":
        return {
            "device": device,
            "oof_folds": 3,
            "n_estimators": 30,
            "max_depth": 4,
            "n_jobs": 2,
        }
    if method_id == "b_dataoob_official":
        return {"n_trees": 200, "n_jobs": 2, "max_depth": 8}
    if method_id == "a_coretab_xgb_ova":
        return {
            "binary_options": {
                "trees_number": 8,
                "sample_percent": 0.20,
                "examples_to_keep": 120,
                "threshold": 1,
                "n_jobs": 2,
            }
        }
    return {}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_toys(*, external_root: Path, seed: int, budget_ratio: float, device: str) -> dict:
    X, y, phases = _toy(seed)
    results = []
    for method_id in MAIN_BASELINE_IDS:
        started = time.perf_counter()
        try:
            selector = build_selector(
                method_id,
                seed,
                external_root=external_root,
                options=_options(method_id, device),
            )
            kwargs = {"phases": phases} if method_id == "c03_phase_class_random" else {}
            selection = selector.select(X, y, budget_ratio, **kwargs)
            row = selection.to_dict()
        except Exception as error:
            row = {
                "method_id": method_id,
                "status": "failed",
                "requested_rows": int(round(len(y) * budget_ratio)),
                "realized_rows": 0,
                "budget_mode": "unknown",
                "diagnostics": {"reason": f"Toy runner lỗi: {error!r}"},
                "timings": {},
            }
        row["wall_seconds"] = time.perf_counter() - started
        # Không bao giờ lưu indices/weights trong evidence public.
        results.append(_safe(row))
    return {
        "schema_version": 1,
        "evidence_kind": "method_fidelity_toy",
        "not_paper_result": True,
        "row_level_artifacts_saved": False,
        "repository_commit": _git_commit(),
        "seed": seed,
        "budget_ratio": budget_ratio,
        "toy_rows": len(y),
        "toy_features": X.shape[1],
        "toy_classes": int(len(np.unique(y))),
        "requested_device": device,
        "python": platform.python_version(),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", default=str(PROJECT / "external" / "repos"))
    parser.add_argument("--output", default=str(PROJECT / "artifacts" / "g10" / "toy_results.json"))
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--budget-ratio", type=float, default=0.20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 0 < args.budget_ratio <= 1:
        raise ValueError("budget-ratio phải nằm trong (0, 1]")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_toys(
        external_root=Path(args.external_root),
        seed=args.seed,
        budget_ratio=args.budget_ratio,
        device=args.device,
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = audit_fidelity_manifest(
        PROJECT / "docs" / "method_fidelity_v2.json",
        PROJECT / "external" / "official_repos.lock.json",
        toy_evidence_path=output,
        require_passed_toys=True,
    )
    print(json.dumps({"output": str(output), "audit": audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
