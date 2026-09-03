"""Tests for soft (penalized) mass balance.

Mass balance is a relaxed accounting constraint rather than exact equality:
reported statistics from different sources are mutually inconsistent, and
forcing residuals to zero hides that inconsistency inside the flow estimates.
These tests pin the two properties that matter — that the tolerance actually
governs how far the solution may depart from balance, and that the hard mode
still reproduces exact equality.
"""

import numpy as np
import pytest

from arcade_mfa_aluminum.graph import (
    MassBalanceSystem,
    soft_mass_balance_block,
)
from arcade_mfa_aluminum.inference.optimization import solve_map


def _conflicting_system():
    """Three flows into one node and one out, with observations that cannot
    balance: 100 + 100 in, 150 out. Exact balance is impossible, so the model
    must decide how to distribute the 50 kt/y discrepancy.
    """
    n = 3
    A = np.array([[1.0, 1.0, -1.0]])          # x0 + x1 - x2 = 0
    b = np.zeros(1)
    y = np.array([100.0, 100.0, 150.0])
    sigma = np.array([10.0, 10.0, 10.0])
    lb, ub = np.zeros(n), np.full(n, 1e4)
    return A, b, y, sigma, lb, ub


def test_soft_block_scales_tolerance_with_node_throughput():
    mb = MassBalanceSystem(A=np.array([[1.0, -1.0], [1.0, -1.0]]),
                           b=np.zeros(2), nodes=[7, 9])
    A, b, s = soft_mass_balance_block(mb, {7: 1000.0, 9: 10.0}, rel_sigma=0.01)
    assert A.shape[0] == 2
    assert np.allclose(s, [10.0, 0.1]), f"tolerance not proportional to throughput: {s}"


def test_soft_block_drops_zero_throughput_nodes():
    """A zero tolerance would be an infinite penalty, so such rows are dropped."""
    mb = MassBalanceSystem(A=np.array([[1.0, -1.0], [1.0, -1.0]]),
                           b=np.zeros(2), nodes=[7, 9])
    A, b, s = soft_mass_balance_block(mb, {7: 1000.0, 9: 0.0}, rel_sigma=0.01)
    assert A.shape[0] == 1 and len(s) == 1
    assert s[0] == pytest.approx(10.0)


def test_invalid_rel_sigma_raises():
    mb = MassBalanceSystem(A=np.array([[1.0, -1.0]]), b=np.zeros(1), nodes=[1])
    with pytest.raises(ValueError, match="rel_sigma"):
        soft_mass_balance_block(mb, {1: 100.0}, rel_sigma=0.0)


def test_hard_mode_forces_zero_residual():
    """The comparison case must still balance exactly."""
    A, b, y, sigma, lb, ub = _conflicting_system()
    res = solve_map(y, sigma, A, b, lb, ub, maxiter=5000)
    assert abs((A @ res.x_map)[0]) < 1e-8


def test_soft_mode_permits_a_residual_and_the_tolerance_governs_it():
    """The whole point: with inconsistent observations, a looser tolerance must
    allow a larger imbalance and hold the flows closer to what was reported.
    """
    A, b, y, sigma, lb, ub = _conflicting_system()
    empty_A, empty_b = np.zeros((0, 3)), np.zeros(0)

    resid, obs_miss = {}, {}
    for tol in (1.0, 10.0, 100.0):
        r = solve_map(
            y, sigma, empty_A, empty_b, lb, ub,
            soft_A=A, soft_b=b, soft_sigma=np.array([tol]), maxiter=5000,
        )
        resid[tol] = abs((A @ r.x_map)[0])
        obs_miss[tol] = float(np.abs(r.x_map - y).sum())

    # Looser tolerance -> larger permitted imbalance.
    assert resid[1.0] < resid[10.0] < resid[100.0], resid
    # ...and correspondingly less distortion of the reported values.
    assert obs_miss[100.0] < obs_miss[10.0] < obs_miss[1.0], obs_miss
    # A tight tolerance should approach the hard solution.
    hard = solve_map(y, sigma, A, b, lb, ub, maxiter=5000)
    tight = solve_map(y, sigma, empty_A, empty_b, lb, ub,
                      soft_A=A, soft_b=b, soft_sigma=np.array([1e-3]), maxiter=5000)
    assert np.allclose(tight.x_map, hard.x_map, atol=1e-2), (
        f"tight soft solve should approach the hard solve:\n"
        f"  soft {tight.x_map}\n  hard {hard.x_map}"
    )


def test_soft_residual_is_not_machine_epsilon():
    """Guards the defect this replaced: residuals of ~1e-13 mean the constraint
    is being enforced exactly, which is what a soft formulation must not do.
    """
    A, b, y, sigma, lb, ub = _conflicting_system()
    r = solve_map(
        y, sigma, np.zeros((0, 3)), np.zeros(0), lb, ub,
        soft_A=A, soft_b=b, soft_sigma=np.array([20.0]), maxiter=5000,
    )
    assert abs((A @ r.x_map)[0]) > 1.0, (
        "a soft constraint with a 20 kt/y tolerance against a 50 kt/y "
        "inconsistency should leave a visible residual"
    )


# ---------------------------------------------------------------------------
# Reported zeros
# ---------------------------------------------------------------------------

def test_zero_observation_is_not_pinned_to_a_negligible_interval():
    """A reported zero usually means 'not measured', not 'provably absent'.

    Pinning such a flow to a ~1e-6 interval asserts more certainty than the data
    support and blocks mass balance from routing anything through it.
    """
    from arcade_mfa_aluminum.pipeline import _build_bounds

    y = np.array([0.0, 100.0])
    lb, ub = _build_bounds(y, {"bounds": {"observed_lower_frac": 0.2,
                                          "observed_upper_frac": 1.8,
                                          "zero_flow_upper": 1.0}})
    assert lb[0] == 0.0 and ub[0] == pytest.approx(1.0)
    assert lb[1] == pytest.approx(20.0) and ub[1] == pytest.approx(180.0)

    # and the allowance is configurable, including back to a hard zero
    _, ub_hard = _build_bounds(y, {"bounds": {"zero_flow_upper": 0.0}})
    assert ub_hard[0] <= 1e-6


def test_zero_observation_sigma_still_reflects_pedigree():
    """With a relative sigma collapsing at zero, an absolute scale takes over --
    but a poorly-evidenced zero must still be looser than a well-evidenced one.
    """
    import pandas as pd
    from arcade_mfa_aluminum.priors.quality_sigma import sigma_for_observations

    obs = pd.DataFrame({
        "source": ["s", "s"], "value": [0.0, 0.0],
        "coverage": [4.0, 1.0], "frequency": [4.0, 1.0], "spatial": [4.0, 1.0],
    })
    sig = sigma_for_observations(
        obs, mapping={"direction": "higher_is_better"}, zero_value_scale=1.0
    )
    assert sig[0] < sig[1], (
        f"a well-evidenced zero should be tighter than a poorly-evidenced one: {sig}"
    )
