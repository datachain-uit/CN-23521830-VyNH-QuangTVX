"""Kiểm tra budget adapter TDBench không bị nhầm với resource gate."""

import numpy as np

from skq_exp.methods.synthetic.tdbench import (
    _classify_resource_exception,
    _count_exact_training_pairs,
    _patch_tdbench_source,
    _plan_tdbench_budget,
    _validate_trajectory_options,
)


def test_kip_adult_uses_largest_source_feasible_budget() -> None:
    y = np.concatenate([
        np.zeros(21_011, dtype=np.int64),
        np.ones(6_665, dtype=np.int64),
    ])
    per_label, diagnostics = _plan_tdbench_budget("s_kip_tdbench", y, 1_384)
    assert per_label == 666
    assert diagnostics["planned_realized_rows"] == 1_332
    assert diagnostics["source_feasibility_limited"] is True
    assert diagnostics["source_max_per_label"] == 666


def test_mtt_keeps_requested_per_class_budget() -> None:
    y = np.concatenate([
        np.zeros(21_011, dtype=np.int64),
        np.ones(6_665, dtype=np.int64),
    ])
    per_label, diagnostics = _plan_tdbench_budget("s_mtt_tdbench", y, 1_384)
    assert per_label == 692
    assert diagnostics["planned_realized_rows"] == 1_384
    assert diagnostics["source_feasibility_limited"] is False


def test_framework_resource_errors_are_not_reported_as_algorithm_failure() -> None:
    assert _classify_resource_exception(RuntimeError("CUDA out of memory")) == "oom"
    assert _classify_resource_exception(RuntimeError("RESOURCE_EXHAUSTED")) == "oom"
    assert _classify_resource_exception(RuntimeError("worker timed out")) == "timeout"
    assert _classify_resource_exception(ValueError("shape mismatch")) is None


def test_empty_training_labels_are_budget_infeasible() -> None:
    per_label, diagnostics = _plan_tdbench_budget(
        "s_kip_tdbench", np.empty(0, dtype=np.int64), 10
    )
    assert per_label == 0
    assert "không có nhãn" in diagnostics["budget_limitation_reason"]


def test_kip_patch_replaces_removed_jax_config_module() -> None:
    source = "import jax\nimport jax.config\nfrom jax.config import config as jax_config\n"
    patched, patch_ids = _patch_tdbench_source("kip", source)
    assert "from jax import config as jax_config" in patched
    assert "import jax.config" not in patched
    assert patch_ids == ["kip_jax_config_import_compat"]


def test_mtt_patch_makes_synthetic_data_and_learning_rate_trainable() -> None:
    source = (
        "random_state = expert_seeds[0]\n"
        "trajectories.append([p.detach().cpu() for p in model.parameters()])\n"
        "        opt_model = optim.SGD\n"
        "trajectories.append([p.detach().cpu() for p in model.parameters()])\n"
        "        all_trajectories.append\n"
        "X_syn = torch.tensor(X[support_idxs]).float().to(device)\n"
        "syn_lr = torch.tensor(lr_teacher).to(device)\n"
        "param_cache = {}\n\n"
        "    for it in range(n_iter):\n"
        "        rng = random.Random(random_state)\n"
        "        start_epoch = rng.randint(0, max_start_epoch)\n"
    )
    patched, patch_ids = _patch_tdbench_source("trajectory_matching", source)
    assert patched.count("requires_grad_(True)") == 2
    assert patched.count("cpu().clone()") == 2
    assert "expert_seeds[i]" in patched
    assert patched.count("random.Random(random_state)") == 1
    assert patch_ids == [
        "mtt_distinct_expert_seeds",
        "mtt_clone_initial_expert_snapshot",
        "mtt_clone_epoch_expert_snapshots",
        "mtt_trainable_synthetic_data",
        "mtt_trainable_learning_rate",
        "mtt_persistent_trajectory_rng",
        "mtt_remove_per_iteration_rng_reset",
    ]


def test_gradient_matching_patch_makes_synthetic_data_trainable() -> None:
    source = "X_syn = torch.tensor(X[support_idxs]).float().to(device)\n"
    patched, patch_ids = _patch_tdbench_source("gradient_matching", source)
    assert patched.count("requires_grad_(True)") == 1
    assert patch_ids == ["gm_trainable_synthetic_data"]


def test_datm_uses_declared_trajectory_patches() -> None:
    source = (
        "random_state = expert_seeds[0]\n"
        "trajectories.append([p.detach().cpu() for p in model.parameters()])\n"
        "        opt_model = optim.SGD\n"
        "trajectories.append([p.detach().cpu() for p in model.parameters()])\n"
        "        all_trajectories.append\n"
        "X_syn = torch.tensor(X[support_idxs]).float().to(device)\n"
        "syn_lr = torch.tensor(lr_teacher).to(device)\n"
        "param_cache = {}\n\n"
        "    for it in range(n_iter):\n"
        "        rng = random.Random(random_state)\n\n"
        "        upper_bound = current_max_start_epoch\n"
    )
    patched, patch_ids = _patch_tdbench_source("datm", source)
    assert patched.count("requires_grad_(True)") == 2
    assert patched.count("cpu().clone()") == 2
    assert patched.count("random.Random(random_state)") == 1
    assert len(patch_ids) == 7
    assert all(item.startswith("datm_") for item in patch_ids)


def test_trajectory_config_rejects_snapshot_out_of_range() -> None:
    parameters = {
        "n_epochs": 5,
        "n_experts": 1,
        "expert_epochs": 2,
        "syn_steps": 1,
        "n_iter": 1,
        "max_start_epoch": 4,
    }
    try:
        _validate_trajectory_options(parameters)
    except ValueError as error:
        assert "max_start_epoch" in str(error)
    else:
        raise AssertionError("Cấu hình vượt trajectory phải bị từ chối")


def test_exact_training_pair_counter_detects_mtt_noop_output() -> None:
    X_train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    y_train = np.array([0, 1, 1], dtype=np.int64)
    X_generated = np.array([[0.0, 1.0], [2.0, 3.1]], dtype=np.float32)
    y_generated = np.array([0, 1], dtype=np.int64)
    assert _count_exact_training_pairs(X_generated, y_generated, X_train, y_train) == 1
