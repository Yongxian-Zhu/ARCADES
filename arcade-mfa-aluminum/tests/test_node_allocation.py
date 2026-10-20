"""Node-allocation formulation: conversion, S17 concentration, MAP and sampling."""

import numpy as np
import pandas as pd
import pytest

from arcade_mfa_aluminum.inference.node_allocation_map import (
    NodeAllocationTerms,
    _chain_grad_z,
    _grad_z,
    _neg_log_post_z,
    _pack,
    _unpack,
    laplace_sample_node_allocation,
    make_objective,
    solve_node_allocation_map,
)
from arcade_mfa_aluminum.node_allocation import (
    DEFAULT_KAPPA_MAX,
    DEFAULT_KAPPA_MIN,
    build_layout,
    dirichlet_from_shares,
    flows_to_node_allocation,
    flows_to_totals_and_shares,
    pedigree_to_concentration,
    quality_index_from_rel_sigma,
)


def _toy_df():
    """Two branching nodes and one pass-through: 1 -> {0,1,2}, 2 -> {3,4}, 3 -> {5}."""
    return pd.DataFrame({
        "flow_idx": [0, 1, 2, 3, 4, 5],
        "from_node": [1, 1, 1, 2, 2, 3],
        "to_node": [2, 3, 9, 3, 9, 9],
    })


def _toy_obs(values, sigmas=None, triple=(4.0, 3.0, 4.0)):
    """Long-form observations in the schema `extract_observations` emits."""
    n = len(values)
    sigmas = [0.05 * abs(v) + 1e-3 for v in values] if sigmas is None else sigmas
    return pd.DataFrame({
        "flow_idx": np.arange(n),
        "source": ["USGS value 1"] * n,
        "value": np.asarray(values, dtype=float),
        "coverage": [triple[0]] * n,
        "frequency": [triple[1]] * n,
        "spatial": [triple[2]] * n,
        "sigma": np.asarray(sigmas, dtype=float),
    })


def _toy_terms(**over):
    lay = build_layout(_toy_df(), 6)
    base = dict(
        layout=lay,
        total_value=np.array([100.0, np.nan, 20.0]),
        total_sigma=np.array([5.0, np.nan, 2.0]),
        share_obs=np.array([0.5, 0.3, 0.2, np.nan, np.nan, np.nan]),
        kappa=np.array([50.0, 10.0, 5.0]),
        resid_flow_idx=np.array([3]),
        resid_value=np.array([12.0]),
        resid_sigma=np.array([1.5]),
        mb_A=np.array([[1.0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0]]),
        mb_sigma=np.array([3.0, 4.0]),
    )
    base.update(over)
    return NodeAllocationTerms(**base)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def test_expand_inverts_the_conversion_exactly():
    """Reparameterisation must lose nothing: x -> (T, s) -> x is the identity.

    If it did not hold, agreement between the two formulations would be
    measuring the conversion rather than the reconciliation.
    """
    lay = build_layout(_toy_df(), 6)
    x = np.array([30.0, 18.0, 12.0, 7.0, 5.0, 42.0])
    T, s = flows_to_totals_and_shares(x, lay)
    assert np.allclose(lay.expand(T, s), x, atol=1e-12), lay.expand(T, s) - x
    for out in lay.out_flows:
        assert abs(s[out].sum() - 1.0) < 1e-12, s[out].sum()


def test_conversion_splits_nodes_by_how_much_is_observed():
    """A partially observed node must keep its per-flow terms.

    Dropping it would silently discard observations, which matters most in the
    sparse target year where most nodes are only partly observed.
    """
    df = _toy_df()
    obs = _toy_obs([30.0, 18.0, 12.0, 7.0, 5.0, 42.0]).iloc[[0, 1, 2, 3, 5]]
    d = flows_to_node_allocation(df, obs, 6)
    status = dict(zip(d.meta.node, d.meta.status))
    assert status[1] == "fully_observed", status
    assert status[2] == "partially_observed", status
    # node 1 is fully observed -> total and shares; node 2 keeps its flow term
    assert np.isfinite(d.total_value[0]) and np.isclose(d.total_value[0], 60.0)
    assert np.allclose(d.share_obs[[0, 1, 2]], [0.5, 0.3, 0.2], atol=1e-12)
    assert 3 in d.resid_flow_idx.tolist(), d.resid_flow_idx
    assert not np.isfinite(d.total_value[1])


def test_unobserved_node_contributes_nothing_but_still_gets_a_layout_slot():
    df = _toy_df()
    obs = _toy_obs([30.0, 18.0, 12.0, 7.0, 5.0, 42.0]).iloc[[0, 1, 2]]
    d = flows_to_node_allocation(df, obs, 6)
    assert dict(zip(d.meta.node, d.meta.status))[2] == "unobserved"
    assert d.layout.n_nodes == 3, d.layout.nodes


# ---------------------------------------------------------------------------
# S15 inversion and S17
# ---------------------------------------------------------------------------

def test_quality_index_inverts_s15_at_the_endpoints():
    assert quality_index_from_rel_sigma(0.05) == pytest.approx(1.0)
    assert quality_index_from_rel_sigma(1.00) == pytest.approx(0.0)
    assert quality_index_from_rel_sigma(0.525) == pytest.approx(0.5, abs=1e-9)
    # outside the range the index saturates rather than going negative
    assert quality_index_from_rel_sigma(2.0) == pytest.approx(0.0)
    assert quality_index_from_rel_sigma(0.001) == pytest.approx(1.0)


def test_concentration_is_monotone_and_hits_both_endpoints():
    """S17 must reward evidence: a better score gives a tighter split."""
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    k = pedigree_to_concentration(q)
    assert np.all(np.diff(k) > 0), k
    assert k[0] == pytest.approx(DEFAULT_KAPPA_MIN)
    assert k[-1] == pytest.approx(DEFAULT_KAPPA_MAX)


def test_concentration_rejects_a_degenerate_range():
    with pytest.raises(ValueError, match="kappa_min"):
        pedigree_to_concentration(0.5, kappa_min=0.0)
    with pytest.raises(ValueError, match="kappa_max"):
        pedigree_to_concentration(0.5, kappa_min=10.0, kappa_max=1.0)


def test_well_scored_flows_give_a_tighter_node_than_poorly_scored_ones():
    df = _toy_df()
    good = flows_to_node_allocation(df, _toy_obs([30.0, 18.0, 12.0, 7.0, 5.0, 42.0],
                                                 sigmas=[0.3, 0.2, 0.1, 0.1, 0.1, 0.4]), 6)
    poor = flows_to_node_allocation(df, _toy_obs([30.0, 18.0, 12.0, 7.0, 5.0, 42.0],
                                                 sigmas=[24.0, 14.0, 10.0, 6.0, 4.0, 34.0]), 6)
    assert good.kappa[0] > poor.kappa[0], (good.kappa[0], poor.kappa[0])


def test_dirichlet_moment_matching_recovers_a_known_concentration():
    """A Dir(kappa*m) sample must map back to roughly kappa."""
    rng = np.random.default_rng(3)
    m = np.array([0.5, 0.3, 0.2])
    kappa = 120.0
    draws = rng.dirichlet(kappa * m, size=40000)
    est = dirichlet_from_shares(draws.mean(axis=0), draws.std(axis=0, ddof=1),
                                kappa_min=1.0, kappa_max=1e4)
    assert est == pytest.approx(kappa, rel=0.05), est


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------

def _fd_grad(f, z, h=1e-6):
    g = np.zeros_like(z)
    for i in range(z.size):
        zp = z.copy(); zp[i] += h
        zm = z.copy(); zm[i] -= h
        g[i] = (f(zp) - f(zm)) / (2 * h)
    return g


def test_gradient_and_hessian_match_finite_differences():
    """The analytic derivatives are load-bearing; a sign slip here would bias
    every result silently rather than failing loudly."""
    terms = _toy_terms()
    fun, grad, hess = make_objective(terms)
    z = np.concatenate([[98.0, 42.0, 19.0], [0.45, 0.33, 0.22, 0.55, 0.45, 1.0]])
    g, fd = grad(z), _fd_grad(fun, z)
    assert np.abs(g - fd).max() < 1e-5 * max(np.abs(fd).max(), 1.0), np.abs(g - fd).max()
    H = hess(z)
    Hfd = np.column_stack([_fd_grad(lambda zz: grad(zz)[k], z) for k in range(z.size)])
    Hfd = 0.5 * (Hfd + Hfd.T)
    assert np.allclose(H, H.T, atol=1e-12)
    assert np.abs(H - Hfd).max() < 1e-4 * max(np.abs(Hfd).max(), 1.0), np.abs(H - Hfd).max()


def test_transform_round_trips_and_has_one_dimension_per_flow():
    lay = build_layout(_toy_df(), 6)
    T = np.array([98.0, 42.0, 19.0])
    s = np.array([0.45, 0.33, 0.22, 0.55, 0.45, 1.0])
    z = _pack(T, s, lay)
    assert z.size == lay.n_flows, (z.size, lay.n_flows)
    T2, s2 = _unpack(z, lay)
    assert np.allclose(T2, T, atol=1e-10), T2 - T
    assert np.allclose(s2, s, atol=1e-12), s2 - s
    for out in lay.out_flows:
        assert abs(s2[out].sum() - 1.0) < 1e-12


def test_transformed_gradients_match_with_and_without_the_jacobian():
    """The MAP and the Laplace expansion point are different problems; both
    gradients must be right, and they must differ by exactly the Jacobian."""
    terms = _toy_terms()
    fun, grad, _ = make_objective(terms)
    lay = terms.layout
    z = _pack(np.array([98.0, 42.0, 19.0]), np.array([0.45, 0.33, 0.22, 0.55, 0.45, 1.0]), lay)

    F = lambda zz: fun(np.concatenate(_unpack(zz, lay)))          # noqa: E731
    g_map = _chain_grad_z(z, lay, grad, include_jacobian=False)
    assert np.abs(g_map - _fd_grad(F, z)).max() < 1e-5 * max(np.abs(_fd_grad(F, z)).max(), 1.0)

    Fj = lambda zz: _neg_log_post_z(zz, lay, fun)                 # noqa: E731
    g_z = _grad_z(z, lay, grad)
    assert np.abs(g_z - _fd_grad(Fj, z)).max() < 1e-5 * max(np.abs(_fd_grad(Fj, z)).max(), 1.0)


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------

def test_map_respects_the_simplex_and_stays_non_negative():
    """Shares are on the simplex by construction, not to a solver tolerance.

    An earlier version renormalised after the solve without clipping first,
    which preserved marginally negative shares and produced negative flows.
    """
    r = solve_node_allocation_map(_toy_terms(), maxiter=4000)
    lay = r.terms.layout
    for out in lay.out_flows:
        assert abs(r.shares[out].sum() - 1.0) < 1e-9, r.shares[out].sum()
    assert (r.shares >= 0).all(), r.shares.min()
    assert (r.flows >= 0).all(), r.flows.min()


def test_map_improves_on_its_starting_point():
    """Guards the failure that a badly scaled constrained solve produced: the
    optimiser ended ABOVE the warm-start objective while appearing to agree
    with it, because it had barely moved."""
    terms = _toy_terms()
    fun, _, _ = make_objective(terms)
    x0 = np.array([30.0, 18.0, 12.0, 7.0, 5.0, 42.0])
    lay = terms.layout
    T0, s0 = flows_to_totals_and_shares(x0, lay)
    start = fun(np.concatenate([T0, s0]))
    r = solve_node_allocation_map(terms, x0_flows=x0, maxiter=8000)
    assert r.objective <= start + 1e-9, (r.objective, start)
    assert r.converged, (r.grad_norm, r.objective)


def test_dirichlet_mode_sits_at_the_observed_share():
    """The mode parameterisation alpha = 1 + kappa*shat is what makes the MAP
    well posed. Under the bare alpha = kappa*shat the density diverges as
    s -> 0 wherever alpha < 1, and the optimiser is driven onto the share floor.
    """
    lay = build_layout(_toy_df(), 6)
    shat = np.array([0.5, 0.3, 0.2, np.nan, np.nan, np.nan])
    terms = NodeAllocationTerms(
        layout=lay, share_obs=shat, kappa=np.array([150.0, 1.0, 1.0]),
        total_value=np.array([100.0, 10.0, 20.0]),
        total_sigma=np.array([0.01, 0.01, 0.01]),
    )
    r = solve_node_allocation_map(terms, maxiter=20000)
    assert np.allclose(r.shares[[0, 1, 2]], [0.5, 0.3, 0.2], atol=5e-3), r.shares[[0, 1, 2]]
    assert r.shares.min() > 1e-6, r.shares.min()


def test_tighter_mass_balance_reduces_the_residual():
    """The soft penalty must actually bind as its tolerance shrinks."""
    A = np.array([[1.0, 0, 0, -1.0, 0, 0]])
    resid = {}
    for rel in (1.0, 0.1, 0.01):
        terms = _toy_terms(mb_A=A, mb_sigma=np.array([rel * 50.0]))
        r = solve_node_allocation_map(terms, maxiter=8000)
        resid[rel] = float(abs(A @ r.flows)[0])
    assert resid[0.01] < resid[0.1] < resid[1.0], resid


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def test_laplace_recovers_a_known_dirichlet_posterior():
    """Bounds how far the Laplace sampler can be trusted.

    With a near-deterministic total and no mass balance, the share posterior is
    exactly Dir(1 + kappa*shat), whose moments are analytic. This is the guard
    against shipping an under-dispersed posterior -- the failure that made the
    originally published intervals ~10x too narrow.
    """
    lay = build_layout(pd.DataFrame({"flow_idx": [0, 1], "from_node": [1, 1],
                                     "to_node": [9, 9]}), 2)
    kappa, shat = 200.0, np.array([0.6, 0.4])
    terms = NodeAllocationTerms(
        layout=lay, share_obs=shat, kappa=np.array([kappa]),
        total_value=np.array([100.0]), total_sigma=np.array([0.01]),
    )
    r = solve_node_allocation_map(terms, maxiter=20000)
    draws = laplace_sample_node_allocation(r, n_draws=20000, seed=7)
    s0 = draws[:, 0] / draws.sum(axis=1)

    a = 1.0 + kappa * shat
    a0 = a.sum()
    mean_exact = a[0] / a0
    sd_exact = np.sqrt(a[0] * (a0 - a[0]) / (a0 ** 2 * (a0 + 1)))

    assert s0.mean() == pytest.approx(mean_exact, abs=3e-3), (s0.mean(), mean_exact)
    assert s0.std(ddof=1) == pytest.approx(sd_exact, rel=0.15), (s0.std(ddof=1), sd_exact)


def test_sampler_draws_are_feasible():
    """Every draw must satisfy the simplex and non-negativity, since they are
    enforced by the parameterisation rather than by rejection."""
    r = solve_node_allocation_map(_toy_terms(), maxiter=4000)
    draws = laplace_sample_node_allocation(r, n_draws=500, seed=1)
    assert (draws >= 0).all(), draws.min()
    lay = r.terms.layout
    for out in lay.out_flows:
        tot = draws[:, out].sum(axis=1)
        sh = draws[:, out] / np.maximum(tot[:, None], 1e-12)
        assert np.abs(sh.sum(axis=1) - 1.0).max() < 1e-9


def test_higher_concentration_tightens_the_share_posterior():
    """The S17 mechanism must do what it claims: more evidence, tighter split."""
    lay = build_layout(pd.DataFrame({"flow_idx": [0, 1], "from_node": [1, 1],
                                     "to_node": [9, 9]}), 2)
    sd = {}
    for kappa in (10.0, 400.0):
        terms = NodeAllocationTerms(
            layout=lay, share_obs=np.array([0.6, 0.4]), kappa=np.array([kappa]),
            total_value=np.array([100.0]), total_sigma=np.array([0.01]),
        )
        r = solve_node_allocation_map(terms, maxiter=20000)
        d = laplace_sample_node_allocation(r, n_draws=4000, seed=5)
        sd[kappa] = float((d[:, 0] / d.sum(axis=1)).std(ddof=1))
    assert sd[400.0] < 0.5 * sd[10.0], sd
