"""Tests for allocation-ratio soft constraints (2017 -> 2022 transfer).

The central claim being tested is that `x_j / sum_k x_k = s_j` can be imposed
as an exactly LINEAR constraint once `s_j` is a constant, and that applying it
softly moves the solution toward the target ratio by an amount the `relaxation`
knob controls -- in both directions.
"""

import numpy as np
import pandas as pd
import pytest

from arcade_mfa_aluminum.allocation import (
    allocation_residual_report,
    build_allocation_constraints,
    compute_allocation_shares,
    load_allocation_shares,
    node_totals_from_flows,
    save_allocation_shares,
)
from arcade_mfa_aluminum.inference.optimization import solve_map


def _toy_df():
    """4 flows: node 1 splits into flows 0,1; node 2 splits into flows 2,3."""
    return pd.DataFrame({
        "flow_idx": [0, 1, 2, 3],
        "from_node": [1, 1, 2, 2],
        "to_node": [10, 11, 12, 13],
        "year": [2017] * 4,
    })


def _draws_with_split(n_draws, split, total=100.0, noise=0.0, seed=0):
    """Draws for the toy graph whose node-1 split is `split` (2-vector)."""
    rng = np.random.default_rng(seed)
    s = np.asarray(split, dtype=float)
    base = np.concatenate([s * total, [40.0, 60.0]])
    out = np.tile(base, (n_draws, 1))
    if noise:
        out = out + rng.normal(0.0, noise, out.shape)
    return out


# ---------------------------------------------------------------------------
# Share estimation
# ---------------------------------------------------------------------------

def test_shares_recover_known_split():
    samples = _draws_with_split(500, [0.7, 0.3], total=100.0, noise=1.0, seed=1)
    shares, w = compute_allocation_shares(samples, _toy_df())

    node1 = shares.to_frame().query("node == 1").sort_values("flow_idx")
    assert np.allclose(node1["share_mean"].to_numpy(), [0.7, 0.3], atol=0.02)
    assert (node1["share_sd"].to_numpy() > 0).all()
    assert w.shape == samples.shape
    assert np.isfinite(w).all()          # every flow here belongs to a branching node


def test_shares_sum_to_one_per_node():
    samples = _draws_with_split(300, [0.55, 0.45], noise=2.0, seed=2)
    shares, _ = compute_allocation_shares(samples, _toy_df())
    per_node = shares.to_frame().groupby("node")["share_mean"].sum()
    assert np.allclose(per_node.to_numpy(), 1.0, atol=1e-9)


def test_single_outflow_nodes_are_excluded():
    df = pd.DataFrame({
        "flow_idx": [0, 1, 2], "from_node": [1, 1, 3],
        "to_node": [10, 11, 12], "year": [2017] * 3,
    })
    samples = np.tile(np.array([70.0, 30.0, 5.0]), (100, 1))
    shares, w = compute_allocation_shares(samples, df)
    assert set(shares.node.tolist()) == {1}, "node 3 has one outflow, carries no ratio"
    assert np.isnan(w[:, 2]).all()


def test_zero_throughput_node_is_skipped_not_nan():
    """A node with ~no throughput has an undefined split; it must be dropped
    rather than emitting NaN shares that poison the constraint matrix.
    """
    df = _toy_df()
    samples = _draws_with_split(200, [0.7, 0.3], noise=0.0, seed=3)
    samples[:, 2:] = 0.0                       # node 2 carries nothing
    shares, _ = compute_allocation_shares(samples, df)
    assert set(shares.node.tolist()) == {1}
    assert np.isfinite(shares.share_mean).all()
    assert np.isfinite(shares.share_sd).all()


def test_save_load_roundtrip(tmp_path):
    samples = _draws_with_split(200, [0.6, 0.4], noise=1.0, seed=4)
    shares, _ = compute_allocation_shares(samples, _toy_df())
    p = tmp_path / "shares.npz"
    save_allocation_shares(shares, str(p))
    back = load_allocation_shares(str(p))
    assert np.array_equal(back.node, shares.node)
    assert np.array_equal(back.flow_idx, shares.flow_idx)
    assert np.allclose(back.share_mean, shares.share_mean)
    assert np.allclose(back.share_sd, shares.share_sd)


# ---------------------------------------------------------------------------
# Constraint construction
# ---------------------------------------------------------------------------

def _toy_constraints(split=(0.7, 0.3), **kw):
    df = _toy_df()
    samples = _draws_with_split(300, split, noise=1.0, seed=5)
    shares, _ = compute_allocation_shares(samples, df)
    totals = node_totals_from_flows(np.array([70.0, 30.0, 40.0, 60.0]), df)
    return df, build_allocation_constraints(df, shares, totals, n_flows=4, **kw)


def test_R_is_zero_exactly_at_the_target_ratio():
    df, con = _toy_constraints(split=(0.7, 0.3))
    s = con.meta.set_index("flow_idx")["target_share"]
    # Use the ESTIMATED shares, not the nominal 0.7/0.3: the estimate carries
    # sampling noise (0.69986), and R is exact with respect to its own target.
    assert abs(s[0] - 0.7) < 0.02

    on_target = np.array([s[0] * 100, s[1] * 100, s[2] * 100, s[3] * 100])
    assert np.abs(con.R @ on_target).max() < 1e-8

    off_target = np.array([0.5 * 100, 0.5 * 100, s[2] * 100, s[3] * 100])
    assert np.abs(con.R @ off_target).max() > 1.0


def test_rows_of_a_node_sum_to_zero():
    """sum_j s_j = 1 makes each node's rows linearly dependent by construction;
    the node contributes rank d-1, not d.
    """
    df, con = _toy_constraints()
    for node in con.meta["node"].unique():
        rows = con.R[con.meta.index[con.meta["node"] == node].to_numpy()]
        assert np.abs(rows.sum(axis=0)).max() < 1e-9


def test_relaxation_scales_tolerance_monotonically():
    _, tight = _toy_constraints(relaxation=1.0, min_share_sigma=1e-6)
    _, loose = _toy_constraints(relaxation=4.0, min_share_sigma=1e-6)
    assert (loose.tau > tight.tau).all()
    assert np.allclose(loose.tau / tight.tau, 4.0, rtol=1e-9)


def test_share_sigma_floor_and_cap_apply():
    _, con = _toy_constraints(relaxation=1.0, min_share_sigma=0.25, max_share_sigma=0.30)
    assert (con.meta["share_sigma"] >= 0.25 - 1e-12).all()
    assert (con.meta["share_sigma"] <= 0.30 + 1e-12).all()


def test_excluded_nodes_are_dropped():
    _, con = _toy_constraints(exclude_nodes=(1,))
    assert set(con.meta["node"].unique()) == {2}


def test_invalid_relaxation_raises():
    with pytest.raises(ValueError, match="relaxation"):
        _toy_constraints(relaxation=0.0)


# ---------------------------------------------------------------------------
# Integration with the MAP solve
# ---------------------------------------------------------------------------

def _solve(con, tau_scale=1.0, y=(50.0, 50.0, 40.0, 60.0)):
    y = np.asarray(y, dtype=float)
    sigma = np.full(4, 10.0)
    A = np.zeros((0, 4))
    return solve_map(
        y, sigma, A, np.zeros(0), np.zeros(4), np.full(4, 1e4),
        soft_A=con.R, soft_sigma=con.tau * tau_scale, maxiter=5000,
    )


def test_tight_constraint_pulls_solution_to_the_target_ratio():
    """Observations say 50/50; the transferred ratio says 70/30. A tight
    tolerance must win, a loose one must not.
    """
    _, con = _toy_constraints(split=(0.7, 0.3), relaxation=1.0, min_share_sigma=1e-4)

    tight = _solve(con, tau_scale=1e-3).x_map
    loose = _solve(con, tau_scale=1e4).x_map

    tight_share = tight[0] / tight[:2].sum()
    loose_share = loose[0] / loose[:2].sum()

    assert abs(tight_share - 0.7) < 0.02, f"tight solve ignored the ratio: {tight_share}"
    assert abs(loose_share - 0.5) < 0.05, f"loose solve was still dragged: {loose_share}"
    assert tight_share > loose_share      # the knob moves it in the right direction


def test_soft_block_keeps_Q_symmetric_psd():
    _, con = _toy_constraints()
    res = _solve(con)
    assert np.allclose(res.Q, res.Q.T)
    assert np.linalg.eigvalsh(res.Q).min() > -1e-8


def test_soft_dimension_mismatch_raises():
    _, con = _toy_constraints()
    with pytest.raises(ValueError, match="soft_sigma"):
        solve_map(
            np.zeros(4), np.ones(4), np.zeros((0, 4)), np.zeros(0),
            np.zeros(4), np.full(4, 1e4),
            soft_A=con.R, soft_sigma=np.ones(con.n_rows + 1),
        )


def test_residual_report_flags_disagreement():
    """When the target year's flows sit at a different split, the report must
    say so via z_vs_tolerance rather than silently absorbing it.
    """
    _, con = _toy_constraints(split=(0.7, 0.3), relaxation=1.0, min_share_sigma=0.01)
    draws = np.tile(np.array([50.0, 50.0, 40.0, 60.0]), (200, 1))   # actually 50/50
    rep = allocation_residual_report(draws, con)
    node1 = rep[rep["node"] == 1]
    assert (node1["z_vs_tolerance"].abs() > 3).any(), (
        "a 50/50 outcome against a 70/30 target should breach 3 sigma"
    )
