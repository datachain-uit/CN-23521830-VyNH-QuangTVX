"""Kiểm kê Kaggle giữ lỗi riêng và không giả định số GPU T4."""

from __future__ import annotations

from skq_exp.experiments.environment import capture_environment


def _runner(command):
    if command[0] == "nvidia-smi" and "--query-gpu=name,memory.total,driver_version" in command:
        return {
            "status": "success", "command": list(command), "returncode": 0,
            "stdout": "Tesla T4, 15360, 550.90\nTesla T4, 15360, 550.90", "stderr": "",
        }
    if command[0] == "df":
        return {
            "status": "failed", "command": list(command), "returncode": 1,
            "stdout": "", "stderr": "fixture failure",
        }
    return {
        "status": "success", "command": list(command), "returncode": 0,
        "stdout": "fixture", "stderr": "",
    }


def test_kaggle_t4_manifest_detects_count_and_keeps_command_failure() -> None:
    payload = capture_environment(
        provider="kaggle-free",
        accelerator_policy="gpu-t4",
        expected_gpu_model="T4",
        session_limit_hours=12,
        available_hours_per_week=None,
        runner=_runner,
    )
    assert payload["environment_status"] == "ready"
    assert payload["gpu_count_detected"] == 2
    assert payload["command_results"]["storage"]["status"] == "failed"
    assert "wall_clock_hours * gpu_count_detected" in payload["accounting"]["allocated_gpu_hours"]


def test_gpu_policy_mismatch_blocks_environment() -> None:
    payload = capture_environment(
        provider="kaggle-free",
        accelerator_policy="gpu-t4",
        expected_gpu_model="A100",
        session_limit_hours=None,
        available_hours_per_week=None,
        runner=_runner,
    )
    assert payload["environment_status"] == "blocked_gpu_mismatch"
