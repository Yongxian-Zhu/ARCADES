"""Smoke test for arcade_mfa_aluminum.inference.optimization -- run with: pytest tests/test_optimization.py"""

import numpy as np

from arcade_mfa_aluminum.inference.optimization import solve_map


def test_solve_map_recovers_observation_when_unconstrained():
    y_obs = np.array([5.0, 10.0, 15.0])
    sigma_obs = np.array([0.5, 0.5, 0.5])
    A = np.zeros((0, 3))
    b = np.zeros(0)
    x_lb = np.zeros(3)
    x_ub = np.full(3, 100.0)

    result = solve_map(y_obs, sigma_obs, A, b, x_lb, x_ub)
    assert result.success
    np.testing.assert_allclose(result.x_map, y_obs, atol=1e-3)


def test_solve_map_respects_mass_balance():
    # flow 0 -> flow1 + flow2  (A: x0 - x1 - x2 = 0)
    y_obs = np.array([10.0, 3.0, 3.0])   # inconsistent: 3+3 != 10
    sigma_obs = np.array([1.0, 1.0, 1.0])
    A = np.array([[1.0, -1.0, -1.0]])
    b = np.array([0.0])
    x_lb = np.zeros(3)
    x_ub = np.full(3, 100.0)

    result = solve_map(y_obs, sigma_obs, A, b, x_lb, x_ub)
    assert result.success
    residual = A @ result.x_map
    np.testing.assert_allclose(residual, 0.0, atol=1e-5)
