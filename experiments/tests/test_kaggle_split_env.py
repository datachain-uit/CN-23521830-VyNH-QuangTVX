"""Kiểm tra orchestration Kaggle hai môi trường mà không cài JAX trong test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kaggle_split_env.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_kaggle_split_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kip_stack_pins_jax_and_cuda_plugin_to_same_version() -> None:
    module = _load_script()
    assert "jax==0.4.38" in module.KIP_DISTRIBUTIONS
    assert "jaxlib==0.4.38" in module.KIP_DISTRIBUTIONS
    assert "jax-cuda12-plugin[with-cuda]==0.4.38" in module.KIP_DISTRIBUTIONS
    assert "jax-cuda12-pjrt==0.4.38" in module.KIP_DISTRIBUTIONS


def test_isolated_panel_and_kip_commands_keep_the_same_config(tmp_path: Path) -> None:
    module = _load_script()
    config = tmp_path / "kaggle.json"
    config.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        config=str(config), dataset="adult_uci2_v1", max_ram_gb=12.0,
        timeout_seconds=7200.0, max_threads=4,
        max_estimated_operations=50_000_000_000,
    )
    panel = module._cell_command(args, "adult_uci2_v1", "c00_full_train", "cat")
    kip = module._cell_command(args, "adult_uci2_v1", "s_kip_tdbench", "mlp")
    assert panel[panel.index("--config") + 1] == kip[kip.index("--config") + 1]
    assert "--method" in kip
    assert panel[panel.index("--learner") + 1] == "cat"
    assert kip[kip.index("--method") + 1] == "s_kip_tdbench"
    assert "--resume" in panel
    assert "--reuse-method-artifacts" in panel


def test_combined_return_code_keeps_running_result_visible() -> None:
    module = _load_script()
    assert module._combined_return_code({"a": 0, "b": -9, "c": 0}) == -9
    assert module._combined_return_code({"a": 0, "b": 0}) == 0


def test_ledger_summary_preserves_kip_failures_and_successes(tmp_path: Path) -> None:
    module = _load_script()
    ledger = tmp_path / "run_ledger.json"
    ledger.write_text(json.dumps([
        {"run_id": "a", "method_id": "c00_full_train", "status": "success"},
        {
            "run_id": "b", "method_id": "s_kip_tdbench", "learner_id": "lr",
            "status": "failed", "reason": "plugin mismatch",
        },
    ]), encoding="utf-8")
    summary = module._summarize_ledger(ledger)
    assert summary["rows"] == 2
    assert summary["status_counts"] == {"failed": 1, "success": 1}
    assert summary["kip_rows"][0]["reason"] == "plugin mismatch"


def test_resource_gate_is_valid_kip_terminal_state() -> None:
    module = _load_script()
    assert module._resource_aware_success([
        {"status": "predicted_oom"},
        {"status": "predicted_timeout"},
    ])
    assert not module._resource_aware_success([{"status": "failed"}])
    assert not module._resource_aware_success([])


def test_method_partition_keeps_config_order_and_protocol_unchanged() -> None:
    module = _load_script()
    config = argparse.Namespace(method_ids=("m1", "m2", "m3", "m4"))
    assert module._select_methods(config, ["m3", "m1"]) == ["m1", "m3"]
    assert module._select_methods(config, []) == ["m1", "m2", "m3", "m4"]


def test_method_partition_rejects_unknown_or_duplicate_methods() -> None:
    module = _load_script()
    config = argparse.Namespace(method_ids=("m1", "m2"))
    for included in (["m3"], ["m1", "m1"]):
        try:
            module._select_methods(config, included)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Partition không hợp lệ vẫn được chấp nhận: {included}")


def test_boundary_timeout_returns_124_instead_of_stopping_panel(monkeypatch) -> None:
    module = _load_script()

    def expire(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["boundary"], timeout=10)

    monkeypatch.setattr(module.subprocess, "run", expire)
    assert module._run(["boundary"], timeout=10) == 124
