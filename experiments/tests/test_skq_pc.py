"""Contract tests cho candidate và ablation SKQ-PC của CQ."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skq_exp.config import ExperimentConfig
from skq_exp.methods import build_selector
from skq_exp.methods.proposed.skq_pc import SKQPCSelector


def _binary_imbalanced_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261017)
    X = rng.normal(size=(160, 10)).astype(np.float32)
    phases = np.repeat(np.arange(4), 40)
    y = np.concatenate([
        np.asarray([0] * 34 + [1] * 6, dtype=np.int64) for _ in range(4)
    ])
    row_ids = np.arange(10_000, 10_160, dtype=np.int64)
    return X, y, phases, row_ids


@pytest.mark.parametrize(
    ("method_id", "extra"),
    [
        ("skq_pc", {}),
        (
            "skq_pc_lrq_sq",
            {
                "query_features": np.linspace(-1, 1, 160, dtype=np.float32)[:, None],
                "query_metadata": {"source": "unit-test OOF stand-in"},
            },
        ),
        ("skq_pc_r", {}),
        ("skq_pc_sqrt", {"allocation_floor": 1}),
        ("x_coretab_mc_cell", {"parent_ids": np.arange(160) % 11}),
    ],
)
def test_skq_pc_family_keeps_exact_budget_and_stratum_mass(
    method_id: str, extra: dict,
) -> None:
    X, y, phases, row_ids = _binary_imbalanced_pool()
    selector = build_selector(
        method_id, 11, n_components=32,
        options={
            "n_components": 32,
            "max_cells_per_stratum": 8,
            **({"allocation_floor": 1} if method_id == "skq_pc_sqrt" else {}),
        },
    )
    result = selector.select(
        X, y, 0.20, phases=phases, row_ids=row_ids, **extra,
    )
    assert result.status == "success", result.diagnostics
    assert result.realized_rows == 32
    assert len(np.unique(result.indices)) == 32
    assert np.isclose(result.weights.sum(), len(y), atol=1e-8)
    assert result.diagnostics["n_strata"] == 8
    assert result.diagnostics["max_stratum_mass_error"] <= 1e-8
    assert result.diagnostics["class_coverage"] == 1.0
    assert result.diagnostics["qp_converged"] is True
    assert result.diagnostics["qp_constraint_valid"] is True
    assert result.diagnostics["passes_over_train"] == 2.0
    assert result.diagnostics["passes_over_train_kind"] == "lower_bound"
    assert all(item["qp_converged"] for item in result.diagnostics["strata"])
    assert all(item["qp_constraint_valid"] for item in result.diagnostics["strata"])


def test_skq_pc_sqrt_uses_registered_allocation_and_rejects_infeasible_floor() -> None:
    X, y, phases, row_ids = _binary_imbalanced_pool()
    valid = build_selector(
        "skq_pc_sqrt", 11, n_components=16,
        options={"allocation_floor": 1, "max_cells_per_stratum": 4},
    ).select(X, y, 0.20, phases=phases, row_ids=row_ids)
    assert valid.status == "success"
    assert valid.diagnostics["allocation_policy"] == "sqrt_floor"
    assert valid.diagnostics["allocation_floor"] == 1

    blocked = build_selector(
        "skq_pc_sqrt", 11, n_components=16,
        options={"allocation_floor": 20, "max_cells_per_stratum": 4},
    ).select(X, y, 0.20, phases=phases, row_ids=row_ids)
    assert blocked.status == "budget_infeasible"
    assert "floor" in blocked.diagnostics["reason"]


def test_skq_pc_random_is_reproducible_for_same_seed() -> None:
    X, y, phases, row_ids = _binary_imbalanced_pool()
    kwargs = {"phases": phases, "row_ids": row_ids}
    first = SKQPCSelector(
        "skq_pc_r", variant="random", seed=29,
        n_components=24, max_cells_per_stratum=6,
    ).select(X, y, 0.20, **kwargs)
    second = SKQPCSelector(
        "skq_pc_r", variant="random", seed=29,
        n_components=24, max_cells_per_stratum=6,
    ).select(X, y, 0.20, **kwargs)
    assert first.status == second.status == "success"
    assert np.array_equal(first.indices, second.indices)
    assert np.allclose(first.weights, second.weights)


def test_skq_pc_does_not_fallback_when_phase_is_missing() -> None:
    X, y, _phases, row_ids = _binary_imbalanced_pool()
    result = SKQPCSelector(
        "skq_pc", variant="standard", seed=11, n_components=16,
    ).select(X, y, 0.20, row_ids=row_ids)
    assert result.status == "blocked"
    assert "không fallback" in result.diagnostics["reason"]


def test_coretab_cell_is_dev_only_and_requires_explicit_pending_ack() -> None:
    source = ExperimentConfig.from_json(
        Path(__file__).parents[1] / "configs" / "s1_screen.json"
    )
    options = {
        "x_coretab_mc_cell": {
            "parent_source": "n02_coretab_xgb_subset",
            "pending_approval_acknowledged": True,
        },
        "skq_pc_lrq_sq": {
            "query_learners": ["lr"], "query_alpha": 0.5,
        },
    }
    dev = replace(
        source,
        stage_id="s1_screen",
        evidence_role="dev_selection",
        method_ids=("skq_pc", "skq_pc_lrq_sq", "skq_pc_r", "x_coretab_mc_cell"),
        method_options=options,
        test_locked=True,
    )
    dev.validate()

    missing_ack = replace(
        dev,
        method_options={
            **options,
            "x_coretab_mc_cell": {"parent_source": "n02_coretab_xgb_subset"},
        },
    )
    with pytest.raises(ValueError, match="pending_approval_acknowledged"):
        missing_ack.validate()
    with pytest.raises(ValueError, match="dev_selection"):
        replace(dev, test_locked=False).validate()
