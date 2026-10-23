"""Chạy panel và KIP nối tiếp trong một Kaggle Save & Run.

Mỗi cặp method–learner chạy trong process riêng. Vì vậy một learner bị kernel
kill/OOM không làm mất các method còn lại; artifact method được dùng lại sau khi
đã kiểm tra identity và ledger được checkpoint sau từng learner. Sau panel,
script đồng bộ JAX/CUDA plugin về 0.4.38 rồi vẫn chạy KIP, kể cả khi một cell
panel trước đó lỗi.

Script luôn aggregate và đóng gói artifact sau cùng, kể cả khi một method
ghi trạng thái failure. Trạng thái thật nằm trong ledger và
``kaggle_execution_summary.json``; không thay method bằng fallback.

Có thể lặp ``--include-method`` để chia cùng một protocol thành nhiều Kaggle
job. Danh sách này chỉ lọc lịch chạy, không sửa config hay protocol hash.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from skq_exp.config import ExperimentConfig


KIP_DISTRIBUTIONS = (
    "jax==0.4.38",
    "jaxlib==0.4.38",
    "jax-cuda12-plugin[with-cuda]==0.4.38",
    "jax-cuda12-pjrt==0.4.38",
    "neural-tangents==0.6.5",
)
KIP_UNINSTALL = (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
)
RESOURCE_TERMINAL_STATUSES = {
    "success", "predicted_oom", "predicted_timeout", "oom", "timeout",
    "storage_limit",
}


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> int:
    print("\n$ " + " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            list(command), env=env, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"Lệnh vượt timeout riêng {timeout}s; tiếp tục các method còn lại.",
            flush=True,
        )
        return 124
    return int(completed.returncode)


def _capture(command: Sequence[str], *, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        list(command), env=env, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return {
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_command(args: argparse.Namespace, *extra: str) -> list[str]:
    command = [
        sys.executable, "-m", "skq_exp.cli", "run",
        "--config", str(Path(args.config).resolve()),
    ]
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    command.extend(extra)
    for flag, value in (
        ("--max-ram-gb", args.max_ram_gb),
        ("--timeout-seconds", args.timeout_seconds),
        ("--max-threads", args.max_threads),
        ("--max-estimated-operations", args.max_estimated_operations),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _cell_command(
    args: argparse.Namespace,
    dataset_id: str,
    method_id: str,
    learner_id: str,
) -> list[str]:
    extra = [
        "--method", method_id,
        "--learner", learner_id,
        "--resume",
        "--reuse-method-artifacts",
    ]
    if not args.dataset:
        extra[0:0] = ["--dataset", dataset_id]
    return _run_command(args, *extra)


def _run_isolated_cells(
    args: argparse.Namespace,
    config: ExperimentConfig,
    *,
    methods: Sequence[str],
    env: dict[str, str] | None = None,
) -> dict[str, int]:
    datasets = [args.dataset] if args.dataset else list(config.dataset_ids)
    return_codes: dict[str, int] = {}
    for dataset_id in datasets:
        for method_id in methods:
            for learner_id in config.learner_ids:
                key = f"{dataset_id}::{method_id}::{learner_id}"
                return_codes[key] = _run(
                    _cell_command(args, dataset_id, method_id, learner_id),
                    env=env,
                )
    return return_codes


def _combined_return_code(return_codes: dict[str, int]) -> int:
    failures = [code for code in return_codes.values() if code != 0]
    return failures[0] if failures else 0


def _select_methods(
    config: ExperimentConfig,
    included_methods: Sequence[str],
) -> list[str]:
    """Lọc lịch chạy nhưng giữ nguyên thứ tự method trong config đã freeze."""
    if not included_methods:
        return list(config.method_ids)
    if len(set(included_methods)) != len(included_methods):
        raise ValueError("--include-method không được lặp cùng một method")
    unknown = sorted(set(included_methods) - set(config.method_ids))
    if unknown:
        raise ValueError(f"Method không có trong config: {unknown}")
    selected = set(included_methods)
    return [method_id for method_id in config.method_ids if method_id in selected]


def _freeze() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return completed.stdout


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() or None


def _distribution_versions() -> dict[str, str | None]:
    names = (
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
        "neural-tangents",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _verify_kip_environment(env: dict[str, str]) -> dict:
    code = (
        "import json, jax, neural_tangents; "
        "print(json.dumps({'jax': jax.__version__, "
        "'devices': [str(v) for v in jax.devices()], "
        "'default_backend': jax.default_backend()}))"
    )
    return _capture([sys.executable, "-c", code], env=env)


def _summarize_ledger(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {"exists": False, "rows": 0, "status_counts": {}, "method_status_counts": {}}
    rows = json.loads(ledger_path.read_text(encoding="utf-8"))
    status_counts = Counter(str(row.get("status", "missing")) for row in rows)
    method_status = Counter(
        f"{row.get('method_id', 'missing')}::{row.get('status', 'missing')}"
        for row in rows
    )
    return {
        "exists": True,
        "rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "method_status_counts": dict(sorted(method_status.items())),
        "kip_rows": [
            {
                key: row.get(key)
                for key in (
                    "run_id", "learner_id", "status", "macro_f1", "mcc", "reason",
                )
            }
            for row in rows if row.get("method_id") == "s_kip_tdbench"
        ],
    }


def _resource_aware_success(rows: list[dict]) -> bool:
    """Resource gate là kết quả hợp lệ; dependency/code failure thì không."""
    return bool(rows) and all(
        str(row.get("status")) in RESOURCE_TERMINAL_STATUSES for row in rows
    )


def _package(artifact_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as handle:
        handle.add(artifact_root, arcname=artifact_root.name)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--max-ram-gb", type=float, default=12.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--max-estimated-operations", type=float, default=50_000_000_000)
    parser.add_argument(
        "--skip-kip-install", action="store_true",
        help="Chỉ dùng khi môi trường KIP đã được pin đúng từ trước.",
    )
    parser.add_argument(
        "--prepare-autocoreset", action="store_true",
        help="Chạy boundary AutoCoreset trước panel; lỗi vẫn được runner ghi vào ledger.",
    )
    parser.add_argument(
        "--autocoreset-timeout-seconds", type=float, default=None,
        help=(
            "Timeout riêng cho boundary AutoCoreset. Khi vượt ngưỡng, chỉ "
            "AutoCoreset dừng và panel vẫn tiếp tục."
        ),
    )
    parser.add_argument(
        "--include-method", action="append", default=[],
        help=(
            "Chỉ chạy method được chỉ định; có thể lặp option. Đây là bộ lọc "
            "điều phối, không thay đổi config hoặc protocol hash."
        ),
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    selected_methods = _select_methods(config, args.include_method)
    artifact_root = Path(args.artifact_root).resolve()
    configured_root = config.paths.artifact_root.resolve()
    if artifact_root != configured_root:
        raise ValueError(
            f"--artifact-root phải khớp config: {artifact_root} != {configured_root}"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (artifact_root / "environment_panel.txt").write_text(_freeze(), encoding="utf-8")

    autocoreset_return_codes: dict[str, int] = {}
    if args.prepare_autocoreset and "n_autocoreset_native" in selected_methods:
        datasets = [args.dataset] if args.dataset else list(config.dataset_ids)
        for dataset_id in datasets:
            for seed in config.selector_seeds:
                key = f"{dataset_id}:ss{seed}"
                autocoreset_return_codes[key] = _run(
                    [
                        sys.executable, "scripts/run_autocoreset_native.py",
                        "--config", str(Path(args.config).resolve()),
                        "--dataset", dataset_id,
                        "--seed", str(seed),
                    ],
                    timeout=args.autocoreset_timeout_seconds,
                )

    # Giữ nguyên config để protocol hash của panel và KIP giống nhau. Chỉ cách
    # điều phối thay đổi: mỗi learner là một process cô lập và có thể resume.
    panel_methods = [
        method_id for method_id in selected_methods
        if method_id != "s_kip_tdbench"
    ]
    panel_return_codes = _run_isolated_cells(
        args, config, methods=panel_methods,
    )
    panel_rc = _combined_return_code(panel_return_codes)

    kip_selected = "s_kip_tdbench" in selected_methods
    uninstall_rc = None
    install_rc = None
    if kip_selected:
        kip_env = dict(os.environ)
        kip_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        kip_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        if not args.skip_kip_install:
            uninstall_rc = _run(
                [sys.executable, "-m", "pip", "uninstall", "-y", *KIP_UNINSTALL],
                env=kip_env,
            )
            install_rc = _run(
                [
                    sys.executable, "-m", "pip", "install", "--no-cache-dir",
                    *KIP_DISTRIBUTIONS,
                ],
                env=kip_env,
            )
        verification = _verify_kip_environment(kip_env)
        (artifact_root / "environment_kip.txt").write_text(
            _freeze(), encoding="utf-8",
        )
        # Dù panel hoặc verify lỗi, vẫn gọi runner để KIP độc lập và ghi failure.
        kip_return_codes = _run_isolated_cells(
            args, config, methods=["s_kip_tdbench"], env=kip_env,
        )
    else:
        verification = {"skipped": True, "reason": "KIP không thuộc partition này"}
        kip_return_codes = {}
    kip_rc = _combined_return_code(kip_return_codes)

    aggregate_rc = _run([
        sys.executable, "-m", "skq_exp.cli", "aggregate",
        "--artifact-root", str(artifact_root),
    ])
    ledger_path = artifact_root / config.experiment_id / "run_ledger.json"
    ledger_summary = _summarize_ledger(ledger_path)
    kip_rows = ledger_summary.get("kip_rows", [])
    kip_success = bool(kip_rows) and all(
        row.get("status") == "success" for row in kip_rows
    )
    kip_resource_aware_success = (
        _resource_aware_success(kip_rows) if kip_selected else True
    )
    summary = {
        "repository_commit": _git_commit(),
        "config": str(Path(args.config).resolve()),
        "experiment_id": config.experiment_id,
        "dataset": args.dataset or list(config.dataset_ids),
        "strategy": "isolated_method_learner_cells_then_pinned_kip",
        "selected_methods": selected_methods,
        "autocoreset_return_codes": autocoreset_return_codes,
        "panel_return_code": panel_rc,
        "panel_return_codes": panel_return_codes,
        "kip_uninstall_return_code": uninstall_rc,
        "kip_install_return_code": install_rc,
        "kip_verification": verification,
        "kip_distribution_versions": _distribution_versions(),
        "kip_run_return_code": kip_rc,
        "kip_return_codes": kip_return_codes,
        "aggregate_return_code": aggregate_rc,
        "kip_all_learners_success": kip_success,
        "kip_resource_aware_success": kip_resource_aware_success,
        "ledger": ledger_summary,
        "elapsed_seconds": time.time() - started,
    }
    (artifact_root / "kaggle_execution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    archive_path = Path(args.archive).resolve()
    _package(artifact_root, archive_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nĐã đóng gói: {archive_path}", flush=True)

    # Archive luôn được tạo; exit code khác 0 giúp Kaggle hiển thị rõ pha lỗi.
    return 0 if (
        panel_rc == 0 and kip_rc == 0 and aggregate_rc == 0
        and kip_resource_aware_success
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
