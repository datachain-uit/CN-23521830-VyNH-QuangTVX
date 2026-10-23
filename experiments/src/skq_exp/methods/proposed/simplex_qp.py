"""File này tìm trọng số không âm có tổng bằng một cho các dòng đã chọn.

Trọng số được tối ưu để trung bình RFF của coreset gần trung bình của nhóm gốc.
Solver dùng projected gradient và trả cả residual; nếu không hội tụ, caller phải
báo failure chứ không âm thầm dùng trọng số khác.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SimplexSolution:
    weights: np.ndarray
    objective: float
    iterations: int
    converged: bool
    sum_violation: float
    min_weight: float
    solver: str


def project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    valid = ordered - cssv / np.arange(1, len(values) + 1) > 0
    rho = np.flatnonzero(valid)[-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(values - theta, 0.0)


def solve_simplex_mean_match(
    selected_Z: np.ndarray,
    target_mean: np.ndarray,
    *,
    max_iter: int = 5000,
    tolerance: float = 1e-8,
    check=None,
) -> SimplexSolution:
    A = np.asarray(selected_Z, dtype=np.float64).T
    target = np.asarray(target_mean, dtype=np.float64)
    m = A.shape[1]
    weights = np.full(m, 1.0 / m)
    spectral = np.linalg.norm(A, ord=2) ** 2
    step = 1.0 / max(2.0 * spectral, 1e-12)
    converged = False
    previous_objective = float("inf")
    stable_steps = 0
    for iteration in range(1, max_iter + 1):
        if check is not None and iteration % 25 == 0:
            check("simplex_qp")
        gradient = 2.0 * A.T @ (A @ weights - target)
        updated = project_simplex(weights - step * gradient)
        residual = A @ updated - target
        objective = float(residual @ residual)
        change_small = np.linalg.norm(updated - weights) <= tolerance
        objective_stable = np.isfinite(previous_objective) and (
            abs(previous_objective - objective) <= tolerance * max(1.0, previous_objective)
        )
        stable_steps = stable_steps + 1 if objective_stable else 0
        if change_small or stable_steps >= 25:
            weights = updated
            converged = True
            break
        weights = updated
        previous_objective = objective
    solver = "projected_gradient"
    if not converged and m > 1:
        # Đây là solver thứ hai cho cùng QP, không phải fallback đổi trọng số.
        optimized = minimize(
            fun=lambda value: float(np.sum((A @ value - target) ** 2)),
            x0=weights,
            jac=lambda value: 2.0 * A.T @ (A @ value - target),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * m,
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0), "jac": lambda value: np.ones_like(value)},
            options={"maxiter": 2000, "ftol": tolerance, "disp": False},
        )
        if optimized.success and np.isfinite(optimized.x).all():
            weights = project_simplex(optimized.x)
            converged = True
            iteration += int(getattr(optimized, "nit", 0))
            solver = "projected_gradient_then_slsqp"
    residual = A @ weights - target
    return SimplexSolution(
        weights=weights,
        objective=float(residual @ residual),
        iterations=iteration,
        converged=converged,
        sum_violation=float(abs(weights.sum() - 1.0)),
        min_weight=float(weights.min()),
        solver=solver,
    )

