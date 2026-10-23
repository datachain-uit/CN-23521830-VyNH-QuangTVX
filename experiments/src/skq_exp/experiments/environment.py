"""Chụp môi trường chạy mà không gộp lỗi lệnh kiểm kê.

QĐ8/QĐ16 yêu cầu báo thiết bị, CPU, RAM, VRAM, phần mềm và giới hạn phiên.
Mỗi lệnh kiểm kê giữ ``success``, ``failed`` hoặc ``missing`` riêng; thiếu một
trường không được suy ra từ trường khác. Policy Kaggle T4 chỉ kiểm tên GPU thực
tế, không giả định số GPU được cấp cho phiên.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ..artifacts import atomic_json, stable_hash


CommandRunner = Callable[[Sequence[str]], dict]


def capture_environment(
    *,
    provider: str,
    accelerator_policy: str,
    expected_gpu_model: str | None,
    session_limit_hours: float | None,
    available_hours_per_week: float | None,
    runner: CommandRunner | None = None,
) -> dict:
    """Trả environment manifest; không thu biến môi trường hay credential."""
    run = runner or _run_command
    commands = {
        "nvidia_smi_inventory": [
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_smi_full": ["nvidia-smi"],
        "cpu": ["lscpu"],
        "memory": ["free", "-h"],
        "storage": ["df", "-h", "/kaggle/working"],
    }
    results = {name: run(command) for name, command in commands.items()}
    gpu_rows = _parse_gpu_inventory(results["nvidia_smi_inventory"])
    model_matches = (
        True if expected_gpu_model is None
        else bool(gpu_rows) and all(
            expected_gpu_model.casefold() in row["name"].casefold() for row in gpu_rows
        )
    )
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost", "torch"):
        try:
            packages[name] = {"status": "success", "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"status": "missing", "version": None}
    cuda = None
    try:
        import torch
        cuda = {
            "torch_cuda_version": torch.version.cuda,
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except (ImportError, RuntimeError) as error:
        cuda = {"status": "unavailable", "reason": type(error).__name__}

    identity = {
        "provider": provider,
        "accelerator_policy": accelerator_policy,
        "expected_gpu_model": expected_gpu_model,
        "gpu_inventory": gpu_rows,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "session_limit_hours": session_limit_hours,
        "available_hours_per_week": available_hours_per_week,
    }
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_id": stable_hash(identity),
        **identity,
        "gpu_count_detected": len(gpu_rows),
        "gpu_model_policy_matches": model_matches,
        "environment_status": "ready" if model_matches else "blocked_gpu_mismatch",
        "command_results": results,
        "accounting": {
            "wall_clock": "seconds measured per run",
            "allocated_gpu_hours": "wall_clock_hours * gpu_count_detected",
            "gpu_utilization": "optional diagnostic; not used to discount allocation",
        },
    }


def write_environment_manifest(output: str | Path, payload: dict) -> Path:
    """Ghi manifest mới và không overwrite evidence môi trường cũ."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"Environment manifest đã tồn tại: {path}")
    atomic_json(path, payload)
    return path


def _run_command(command: Sequence[str]) -> dict:
    try:
        completed = subprocess.run(
            list(command), check=False, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return {"status": "missing", "command": list(command), "returncode": None,
                "stdout": "", "stderr": "command not found"}
    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _parse_gpu_inventory(result: dict) -> list[dict]:
    if result.get("status") != "success":
        return []
    rows = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, memory_mib, driver = parts
        try:
            memory = int(memory_mib)
        except ValueError:
            continue
        rows.append({"name": name, "memory_total_mib": memory, "driver_version": driver})
    return rows
