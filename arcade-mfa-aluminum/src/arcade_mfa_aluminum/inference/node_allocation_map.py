"""
arcade_mfa_aluminum.inference.node_allocation_map
-------------------------------------------------
MAP estimation and posterior sampling for the node-allocation formulation.

The objective is not quadratic, so `inference.optimization.solve_map` cannot be
reused: mass balance is bilinear in (T, s) and the Dirichlet terms are
logarithmic. This module carries its own objective, an analytic gradient, and a
Laplace posterior in an unconstrained transform.

Negative log posterior
----------------------
With x = expand(T, s)::

    0.5 sum_p (T_p - That_p)^2 / sigma_p^2                 node totals
  - sum_p sum_j (kappa_p shat_j) log s_j                   Dirichlet, S17
  + 0.5 sum_j (T_p s_j - y_j)^2 / sigma_j^2                partially observed nodes
  + 0.5 (A x)^T W_mb (A x)                                 soft mass balance
  + 0.5 sum_p (T_p - mu_p)^2 / tau_p^2                     transferred prior, S18
  - sum_p sum_j (kappa0_p m_j) log s_j                     Dirichlet prior

Every term is optional; a run with no prior and no aggregates simply drops the
corresponding blocks.

Sampling
--------
The constrained space (T >= 0, shares on per-node simplices) is mapped to an
unconstrained one of exactly `n_flows` dimensions -- log for the totals,
additive log-ratio for each simplex -- the Hessian is taken there by
finite-differencing the analytic gradient, and draws are pushed back through the
inverse transform. The Jacobian correction is included, so the Laplace
approximation targets the pushforward of the posterior rather than of the
objective.

This is a weaker sampler than the flow formulation's truncated Gibbs and is
intended for cross-checking. `tests/test_node_allocation.py` bounds how far it
can be trusted by comparing it against an analytically tractable case.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from arcade_mfa_aluminum.node_allocation import NodeAllocationLayout, flows_to_totals_and_shares

#: Shares are bounded away from zero so that log s stays finite.
SHARE_FLOOR = 1e-9


@dataclass
class NodeAllocationTerms:
    """Everything the objective needs, already aligned to the layout."""

    layout: NodeAllocationLayout
    total_value: np.ndarray = None       # (n_nodes,) NaN where absent
    total_sigma: np.ndarray = None
    share_obs: np.ndarray = None         # (n_flows,) NaN where absent
    kappa: np.ndarray = None             # (n_nodes,)
    resid_flow_idx: np.ndarray = None
    resid_value: np.ndarray = None
    resid_sigma: np.ndarray = None
    mb_A: np.ndarray = None              # (n_constraints, n_flows)
    mb_sigma: np.ndarray = None
    prior_total_mean: np.ndarray = None  # (n_nodes,)
    prior_total_sd: np.ndarray = None
    prior_share_mean: np.ndarray = None  # (n_flows,)
    prior_kappa: np.ndarray = None       # (n_nodes,)


@dataclass
class NodeAllocationResult:
    totals: np.ndarray
    shares: np.ndarray
    flows: np.ndarray
    success: bool
    message: str
    n_iter: int
    objective: float
    grad_norm: float = np.nan        # max |dF/dz| at the solution
    terms: NodeAllocationTerms = None
    samples: np.ndarray = field(default=None)   # (n_draws, n_flows) in FLOW space

    @property
    def converged(self) -> bool:
        """Stationarity relative to the objective scale.

        L-BFGS-B reports success only on an absolute gradient tolerance, which
        is unreachable here: the objective is order 1e3, so a gradient component
        of 1e-2 is already stationary to nine significant figures. This measures
        what actually matters.
        """
        return bool(self.grad_norm <= 1e-4 * max(1.0, abs(self.objective)))


# ---------------------------------------------------------------------------
# objective and gradient, in (T, s) space
# ---------------------------------------------------------------------------

def _dirichlet_alpha(terms: NodeAllocationTerms):
    """Per-flow Dirichlet exponent ``alpha_j - 1``, summed over likelihood and prior.

    **Mode parameterisation.** A Dirichlet is written here as
    ``alpha_j = 1 + kappa_p * shat_j``, so the exponent returned is simply
    ``kappa_p * shat_j``.

    This is not cosmetic. Under the bare ``alpha_j = kappa_p * shat_j`` the
    density is unbounded as ``s_j -> 0`` wherever ``alpha_j < 1``, which happens
    for any small observed share, and the MAP is then degenerate at the simplex
    boundary: the optimiser drives those shares onto the floor and the gradient
    diverges. Adding one puts the mode of ``Dir(1 + kappa * shat)`` exactly at
    ``shat`` -- since ``mode_j = (alpha_j - 1) / (sum_j alpha_j - k)
    = kappa shat_j / kappa = shat_j`` -- and keeps the exponent non-negative, so
    the term is convex in each ``log s_j`` and repels shares from zero rather
    than attracting them to it.

    ``kappa`` then reads as concentration *in excess of uniform*: the total
    concentration is ``k + kappa_p``.

    Likelihood and prior combine additively in the exponent. Returns None when
    neither is present.
    """
    layout = terms.layout
    acc = np.zeros(layout.n_flows)
    used = False
    if terms.share_obs is not None and terms.kappa is not None:
        for i, out in enumerate(layout.out_flows):
            so = terms.share_obs[out]
            if len(out) > 1 and np.isfinite(so).all():
                acc[out] += terms.kappa[i] * so
                used = True
    if terms.prior_share_mean is not None and terms.prior_kappa is not None:
        for i, out in enumerate(layout.out_flows):
            pm = terms.prior_share_mean[out]
            if len(out) > 1 and np.isfinite(pm).all():
                acc[out] += terms.prior_kappa[i] * pm
                used = True
    return acc if used else None


def make_objective(terms: NodeAllocationTerms):
    """Return (fun, grad, hess) over the concatenated vector z = [T, s]."""
    layout = terms.layout
    n_nodes, n_flows = layout.n_nodes, layout.n_flows
    pos = layout.flow_node_pos
    alpha_m1 = _dirichlet_alpha(terms)

    tv, ts = terms.total_value, terms.total_sigma
    tot_mask = (np.isfinite(tv) & np.isfinite(ts)) if tv is not None else np.zeros(n_nodes, bool)
    tot_w = np.zeros(n_nodes)
    if tot_mask.any():
        tot_w[tot_mask] = 1.0 / np.maximum(ts[tot_mask] ** 2, 1e-12)

    has_resid = terms.resid_flow_idx is not None and len(terms.resid_flow_idx) > 0
    if has_resid:
        r_idx = np.asarray(terms.resid_flow_idx, dtype=int)
        r_val = np.asarray(terms.resid_value, dtype=float)
        r_w = 1.0 / np.maximum(np.asarray(terms.resid_sigma, dtype=float) ** 2, 1e-12)

    has_mb = terms.mb_A is not None and terms.mb_A.size and terms.mb_A.shape[0] > 0
    if has_mb:
        mb_A = np.asarray(terms.mb_A, dtype=float)
        mb_w = 1.0 / np.maximum(np.asarray(terms.mb_sigma, dtype=float) ** 2, 1e-12)

    has_tprior = terms.prior_total_mean is not None and terms.prior_total_sd is not None
    if has_tprior:
        p_mu = np.asarray(terms.prior_total_mean, dtype=float)
        p_w = 1.0 / np.maximum(np.asarray(terms.prior_total_sd, dtype=float) ** 2, 1e-12)

    def split(z):
        return z[:n_nodes], z[n_nodes:]

    def fun(z):
        T, s = split(z)
        s = np.maximum(s, SHARE_FLOOR)
        val = 0.0
        if tot_mask.any():
            d = T - np.where(tot_mask, tv, 0.0)
            val += 0.5 * float(np.sum(tot_w * d * d))
        if alpha_m1 is not None:
            val -= float(np.sum(alpha_m1 * np.log(s)))
        x = T[pos] * s
        if has_resid:
            e = x[r_idx] - r_val
            val += 0.5 * float(np.sum(r_w * e * e))
        if has_mb:
            r = mb_A @ x
            val += 0.5 * float(np.sum(mb_w * r * r))
        if has_tprior:
            d = T - p_mu
            val += 0.5 * float(np.sum(p_w * d * d))
        return val

    def grad(z):
        T, s = split(z)
        s = np.maximum(s, SHARE_FLOOR)
        dT = np.zeros(n_nodes)
        ds = np.zeros(n_flows)
        if tot_mask.any():
            dT += tot_w * (T - np.where(tot_mask, tv, 0.0))
        if alpha_m1 is not None:
            ds -= alpha_m1 / s
        x = T[pos] * s
        gx = np.zeros(n_flows)
        if has_resid:
            e = x[r_idx] - r_val
            np.add.at(gx, r_idx, r_w * e)
        if has_mb:
            gx += mb_A.T @ (mb_w * (mb_A @ x))
        if has_resid or has_mb:
            # chain rule through x_j = T_{p(j)} * s_j
            np.add.at(dT, pos, gx * s)
            ds += gx * T[pos]
        if has_tprior:
            dT += p_w * (T - p_mu)
        return np.concatenate([dT, ds])

    # flow -> node indicator, used to fold the per-flow chain rule into node blocks
    M = np.zeros((n_flows, n_nodes))
    M[np.arange(n_flows), pos] = 1.0

    def hess(z):
        """Analytic Hessian.

        trust-constr falls back to a quasi-Newton approximation without this and
        exhausts its evaluation budget on a problem this size -- the same
        failure the flow formulation hit before its Hessian was supplied.

        Under the mode parameterisation the Dirichlet block is
        kappa_p shat_j / s_j^2 >= 0, so that term is convex. Indefiniteness can
        still arise from the bilinear mass-balance cross terms; trust-constr
        handles it.
        """
        T, s = split(z)
        s = np.maximum(s, SHARE_FLOOR)
        H = np.zeros((n_nodes + n_flows, n_nodes + n_flows))
        Tb, Sb = slice(0, n_nodes), slice(n_nodes, n_nodes + n_flows)

        if tot_mask.any():
            H[Tb, Tb] += np.diag(tot_w)
        if alpha_m1 is not None:
            H[Sb, Sb] += np.diag(alpha_m1 / (s * s))
        if has_tprior:
            H[Tb, Tb] += np.diag(p_w)

        Tf = T[pos]
        if has_resid:
            w_full = np.zeros(n_flows)
            y_full = np.zeros(n_flows)
            w_full[r_idx] = r_w
            y_full[r_idx] = r_val
            # d2/dT_p2 = sum_j w s_j^2 ; d2/ds_j2 = w T_p^2 ; cross = w(2 T s - y)
            H[Tb, Tb] += np.diag(M.T @ (w_full * s * s))
            H[Sb, Sb] += np.diag(w_full * Tf * Tf)
            cross = w_full * (2.0 * Tf * s - y_full)
            C = M * cross[:, None]          # (n_flows, n_nodes)
            H[Tb, Sb] += C.T
            H[Sb, Tb] += C

        if has_mb:
            x = Tf * s
            r = mb_A @ x
            J_T = (mb_A * s[None, :]) @ M           # (n_con, n_nodes)
            J_s = mb_A * Tf[None, :]                # (n_con, n_flows)
            J = np.hstack([J_T, J_s])
            H += J.T @ (mb_w[:, None] * J)          # Gauss-Newton part
            # second-order: d2 r_c / dT_p ds_j = A[c, j] for j in O(p)
            gx = mb_A.T @ (mb_w * r)                # (n_flows,)
            C = M * gx[:, None]
            H[Tb, Sb] += C.T
            H[Sb, Tb] += C

        return 0.5 * (H + H.T)

    return fun, grad, hess


# ---------------------------------------------------------------------------
# unconstrained transform
#
# The constrained problem is badly scaled for a bounded, equality-constrained
# solver: shares span four orders of magnitude and the Dirichlet curvature
# kappa/s reaches ~1e6 on the smallest of them, so trust-constr takes a large
# early excursion and then crawls back. Reparameterising removes the simplex
# and the positivity constraints entirely and puts the small shares on a log
# scale, which is both better conditioned and cheaper.
#
#   u_p = log T_p                                  (totals, positivity implicit)
#   v   = additive log-ratio within each simplex   (sum-to-one implicit)
#
# The transform is a bijection onto exactly n_flows dimensions, so minimising
# f(g(z)) over an unconstrained z finds the same point as minimising f over the
# constrained set.
# ---------------------------------------------------------------------------

def _pack(T, s, layout):
    """(T, s) -> z: log totals, then additive log-ratio per simplex."""
    u = np.log(np.maximum(T, 1e-12))
    v = []
    for out in layout.out_flows:
        if len(out) > 1:
            sv = np.maximum(s[out], SHARE_FLOOR)
            v.append(np.log(sv[:-1] / sv[-1]))
    return np.concatenate([u] + v) if v else u


def _unpack(z, layout):
    """z -> (T, s), the inverse of `_pack`. Shares are exactly on the simplex."""
    n = layout.n_nodes
    T = np.exp(z[:n])
    s = np.zeros(layout.n_flows)
    k = n
    for out in layout.out_flows:
        m = len(out)
        if m == 1:
            s[out[0]] = 1.0
            continue
        vv = z[k:k + m - 1]
        k += m - 1
        shift = max(float(vv.max()), 0.0)          # stabilised softmax
        e = np.exp(vv - shift)
        last = np.exp(-shift)
        den = e.sum() + last
        s[out[:-1]] = e / den
        s[out[-1]] = last / den
    return T, s


def _chain_grad_z(z, layout, grad, *, include_jacobian: bool):
    """Gradient in z-space by the chain rule.

    `include_jacobian` adds the derivative of `-log|det J|`. Use False to find
    the MAP of the ORIGINAL (T, s) posterior -- the quantity comparable with the
    flow formulation -- and True to find the mode of the transformed density,
    which is what a Laplace approximation in z-space must expand about.
    """
    n = layout.n_nodes
    T, s = _unpack(z, layout)
    g = grad(np.concatenate([T, s]))
    dT, ds = g[:n], g[n:]

    gu = dT * T
    if include_jacobian:
        gu = gu - 1.0
    gv = []
    for out in layout.out_flows:
        m = len(out)
        if m == 1:
            continue
        sv, dsv = s[out], ds[out]
        dot = float(np.sum(dsv * sv))
        # ds_i/dv_n = s_i (delta_in - s_n)  =>  df/dv_n = s_n (df/ds_n - dot)
        core = sv[:-1] * (dsv[:-1] - dot)
        if include_jacobian:
            core = core + (m * sv[:-1] - 1.0)
        gv.append(core)
    return np.concatenate([gu] + gv) if gv else gu


def _neg_log_post_z(z, layout, fun):
    """Transformed negative log posterior, including the Jacobian.

    log|det J| = sum_p log T_p + sum_p sum_{j in O(p)} log s_j. Omitting it would
    give a Laplace approximation to the transformed *objective* rather than to
    the transformed posterior, which is a different and wrong thing.
    """
    T, s = _unpack(z, layout)
    logdet = float(np.sum(np.log(np.maximum(T, 1e-300))))
    for out in layout.out_flows:
        if len(out) > 1:
            logdet += float(np.sum(np.log(np.maximum(s[out], 1e-300))))
    return fun(np.concatenate([T, s])) - logdet


def _grad_z(z, layout, grad):
    """Gradient of `_neg_log_post_z`. Kept as a named helper for the tests."""
    return _chain_grad_z(z, layout, grad, include_jacobian=True)


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------

def solve_node_allocation_map(
    terms: NodeAllocationTerms,
    *,
    x0_flows: np.ndarray = None,
    maxiter: int = 20000,
    verbose: int = 0,
) -> NodeAllocationResult:
    """MAP over node totals and allocation shares.

    Solved unconstrained in the transform above, so the returned shares sit on
    the simplex and the totals are positive by construction rather than to a
    solver tolerance.

    `x0_flows` warm-starts from a flow-space solution. The bilinear
    mass-balance term makes the objective non-convex, so the starting point
    matters; beginning from the flow-based answer keeps the comparison between
    the two formulations meaningful rather than contrasting unrelated optima.
    `objective` is reported so a poor optimum is visible.
    """
    layout = terms.layout
    n_nodes, n_flows = layout.n_nodes, layout.n_flows
    fun, grad, _hess = make_objective(terms)

    if x0_flows is not None:
        T0, s0 = flows_to_totals_and_shares(np.asarray(x0_flows, dtype=float), layout)
    else:
        T0 = np.where(np.isfinite(terms.total_value), terms.total_value, 1.0) \
            if terms.total_value is not None else np.ones(n_nodes)
        s0 = np.zeros(n_flows)
        for out in layout.out_flows:
            s0[out] = 1.0 / len(out)
    T0 = np.maximum(T0, 1e-6)
    s0 = np.clip(s0, SHARE_FLOOR, 1.0)
    for out in layout.out_flows:
        s0[out] = s0[out] / s0[out].sum()
    z0 = _pack(T0, s0, layout)

    def F(z):
        T, s = _unpack(z, layout)
        return fun(np.concatenate([T, s]))

    def G(z):
        return _chain_grad_z(z, layout, grad, include_jacobian=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(F, z0, jac=G, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 10 * maxiter,
                                "ftol": 1e-15, "gtol": 1e-12, "iprint": verbose - 1})

    T, s = _unpack(res.x, layout)
    obj = float(F(res.x))
    if obj > float(F(z0)) + 1e-9:
        warnings.warn(
            f"solve_node_allocation_map: the optimiser ended above its starting "
            f"objective ({obj:.6g} vs {F(z0):.6g}); the warm start is being reported "
            f"instead. This means the solve failed, not that the start was optimal.",
            stacklevel=2,
        )
        T, s = _unpack(z0, layout)
        obj = float(F(z0))

    gnorm = float(np.abs(G(_pack(T, s, layout))).max())
    return NodeAllocationResult(
        totals=T, shares=s, flows=layout.expand(T, s),
        success=bool(res.success), message=str(res.message),
        n_iter=int(getattr(res, "nit", -1)), objective=obj, grad_norm=gnorm, terms=terms,
    )


# ---------------------------------------------------------------------------
# Laplace posterior
# ---------------------------------------------------------------------------

def laplace_sample_node_allocation(
    result: NodeAllocationResult,
    *,
    n_draws: int = 4000,
    seed: int = 42,
    ridge: float = 1e-8,
    refine: bool = True,
    maxiter: int = 20000,
    max_sd_z: float = 10.0,
) -> np.ndarray:
    """Draw from a Laplace approximation in the unconstrained transform.

    The expansion point is the mode of the *transformed* density, which differs
    from the image of the (T, s) MAP by the Jacobian term; `refine` re-optimises
    to find it. Returns (n_draws, n_flows) in FLOW space, directly comparable
    with the flow formulation's `posterior_samples.npy`.
    """
    layout = result.terms.layout
    fun, grad, _hess = make_objective(result.terms)
    z0 = _pack(result.totals, result.shares, layout)

    if refine:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda z: _neg_log_post_z(z, layout, fun), z0,
                           jac=lambda z: _grad_z(z, layout, grad),
                           method="L-BFGS-B",
                           options={"maxiter": maxiter, "maxfun": 10 * maxiter,
                                    "ftol": 1e-14, "gtol": 1e-10})
        if np.all(np.isfinite(res.x)):
            z0 = res.x

    d = z0.size
    step = 1e-5 * np.maximum(np.abs(z0), 1.0)
    H = np.empty((d, d))
    for i in range(d):
        zp = z0.copy(); zp[i] += step[i]
        zm = z0.copy(); zm[i] -= step[i]
        H[:, i] = (_grad_z(zp, layout, grad) - _grad_z(zm, layout, grad)) / (2 * step[i])
    H = 0.5 * (H + H.T)

    w, V = np.linalg.eigh(H)
    floor = ridge * max(float(w.max()), 1.0)
    w = np.maximum(w, floor)
    sd = 1.0 / np.sqrt(w)

    # A flat direction has no meaningful Laplace spread: 1/sqrt(w) runs to
    # hundreds, and since the totals are sampled in log space exp() then
    # overflows to absurd magnitudes. Cap the step instead of emitting numbers
    # that are worse than useless -- and say so, because a capped direction is
    # unidentified, not precisely known.
    n_capped = int((sd > max_sd_z).sum())
    if n_capped:
        warnings.warn(
            f"laplace_sample_node_allocation: {n_capped}/{d} directions are flat at the "
            f"mode (implied sd up to {sd.max():.3g} in transformed units) and were capped "
            f"at {max_sd_z}. Those quantities are NOT identified by the data; their "
            f"reported spread is an artefact of the cap and must not be quoted as a "
            f"credible interval.",
            stacklevel=2,
        )
    sd = np.minimum(sd, max_sd_z)

    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_draws, d))
    Z = z0 + (g * sd) @ V.T

    out = np.empty((n_draws, layout.n_flows))
    for i in range(n_draws):
        T, s = _unpack(Z[i], layout)
        out[i] = layout.expand(T, s)
    return out
