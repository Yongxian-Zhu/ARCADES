"""
arcade_mfa_aluminum.inference.sampling
--------------------------------------
Two posterior sampling backends:

1. `truncated_gibbs_sample_nullspace` (default) -- samples the Gaussian
   posterior restricted to BOTH the mass-balance subspace and the box bounds,
   so every draw is feasible by construction. This is the method whose output
   should be reported.

2. `laplace_sample_nullspace` -- the original scheme, which draws from the
   unconstrained Laplace approximation, clips to the bounds and then projects
   onto the mass-balance subspace. Because the projection runs last it can move
   a draw back outside the box, so the draws it returns are not guaranteed to
   be feasible. Retained only to reproduce results from the original notebooks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from arcade_mfa_aluminum.graph import nullspace, project_to_mass_balance


@dataclass
class PosteriorSamples:
    samples: np.ndarray   # (n_chains, n_draws, n_flows)
    x_map: np.ndarray
    method: str
    # Sampler behaviour, reported as run diagnostics.
    stats: dict = None


def laplace_sample_nullspace(
    x_map: np.ndarray,
    Q: np.ndarray,
    A_balance: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    *,
    n_chains: int = 4,
    n_draws: int = 2000,
    seed: int = 42,
) -> PosteriorSamples:
    """Laplace approximation: N(x_map, (Z^T Q Z)^-1) restricted to the
    mass-balance nullspace, then clipped to bounds and re-projected.

    Note that the final projection can move a clipped draw back outside the
    box: the returned draws are not guaranteed to satisfy the bounds. Prefer
    `truncated_gibbs_sample_nullspace`.
    """
    n_vars = x_map.shape[0]
    Z = nullspace(A_balance)
    Q_eff = Z.T @ Q @ Z

    try:
        L_eff = np.linalg.cholesky(Q_eff + 1e-12 * np.eye(Q_eff.shape[0]))
        chol_like = True
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Q_eff)
        w = np.maximum(w, 1e-12)
        L_eff = V @ np.diag(np.sqrt(w))
        chol_like = False

    rng = np.random.default_rng(seed)
    samples = np.empty((n_chains, n_draws, n_vars), dtype=float)

    for ci in range(n_chains):
        for di in range(n_draws):
            z0 = rng.standard_normal(Q_eff.shape[0])
            # Q_eff = L_eff @ L_eff.T, so a draw from N(0, Q_eff^-1) is the
            # solution of L_eff.T g = z0: cov(g) = L_eff^-T L_eff^-1 = Q_eff^-1.
            # Solving against L_eff itself would give (L_eff.T L_eff)^-1, which
            # has the right spectrum but the wrong eigenvectors -- i.e. wrong
            # per-flow variances.
            if chol_like:
                g = np.linalg.solve(L_eff.T, z0)
            else:
                g = np.linalg.lstsq(L_eff.T, z0, rcond=None)[0]
            x_s = x_map + Z @ g
            x_s = np.clip(x_s, x_lb, x_ub)
            x_s = project_to_mass_balance(x_s, A_balance)
            samples[ci, di, :] = x_s

    return PosteriorSamples(samples=samples, x_map=x_map, method="laplace_nullspace")


# ---------------------------------------------------------------------------
# Truncated-normal (constrained) sampler
# ---------------------------------------------------------------------------

def _standard_truncnorm_tail(a: float, b: float, rng, max_tries: int = 500) -> float:
    """Standard normal truncated to [a, b] with a > 0, via Robert (1995)
    translated-exponential rejection.

    Used where the inverse-CDF route loses all precision because both
    Phi(a) and Phi(b) round to the same double.
    """
    alpha = 0.5 * (a + np.sqrt(a * a + 4.0))
    for _ in range(max_tries):
        z = a + rng.exponential(1.0 / alpha)
        if z > b:
            continue
        if rng.random() <= np.exp(-0.5 * (z - alpha) ** 2):
            return z
    return a  # pathologically thin interval: sit on the boundary


def _draw_truncnorm(mu: float, sd: float, lo: float, hi: float, rng) -> float:
    """One draw from N(mu, sd^2) truncated to [lo, hi]."""
    if not (hi > lo):
        return float(min(max(mu, lo), hi))

    a = (lo - mu) / sd
    b = (hi - mu) / sd
    pa = ndtr(a)
    pb = ndtr(b)

    if pb - pa > 1e-9:
        p = pa + rng.random() * (pb - pa)
        p = min(max(p, 1e-16), 1.0 - 1e-16)
        return mu + sd * ndtri(p)

    # Interval sits far out in a tail: inverse-CDF would return garbage.
    if a > 0:
        z = _standard_truncnorm_tail(a, b, rng)
    elif b < 0:
        z = -_standard_truncnorm_tail(-b, -a, rng)
    else:
        z = min(max(0.0, a), b)
    return mu + sd * z


def truncated_gibbs_sample_nullspace(
    x_map: np.ndarray,
    Q: np.ndarray,
    A_balance: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    *,
    n_chains: int = 4,
    n_draws: int = 2000,
    n_tune: int = 500,
    thin: int = 1,
    n_direction_moves: int = 16,
    seed: int = 42,
) -> PosteriorSamples:
    """Gibbs sampler for the Gaussian posterior restricted to BOTH the
    mass-balance subspace and the box bounds.

    Why this exists
    ---------------
    Clipping to the box and then projecting onto {x : A x = b} cannot satisfy
    both constraint sets: the projection runs last and moves the draw back off
    the box. Where bounds are wide and clipping is frequently active -- as for
    the many unobserved flows in a sparse year -- a large fraction of draws end
    up infeasible, including physically impossible negative flows.

    How this works
    --------------
    Write x = x_map + W u, where the columns of W span null(A_balance) and are
    whitened so that u has precision I. Every u then satisfies A x = A x_map = b
    exactly, so mass balance holds by construction and never needs re-imposing,
    and the target in u is a standard normal truncated to the polytope
    {u : x_lb <= x_map + W u <= x_ub}.

    Each sweep updates one coordinate of u at a time. Because the whitened
    target is isotropic, the conditional for u_j is simply N(0, 1) truncated to
    the interval that keeps every component of x inside its box -- no precision
    terms enter the conditional mean. Both constraint sets therefore hold at
    every step, exactly, with no projection and no clipping.

    Each sweep also performs `n_direction_moves` hit-and-run updates along
    random directions. Axis-aligned moves alone mix poorly in the corners of the
    feasible polytope, and additional burn-in and thinning does not compensate.
    In whitened coordinates the target is isotropic, so along any unit direction
    d the conditional is simply N(-u.d, 1) truncated to the feasible segment: a
    global move for the cost of one matrix-vector product. Both kernels leave
    the same target invariant, so alternating them is valid MCMC.

    Unlike the clip-and-project scheme this targets the correct constrained
    distribution rather than merely landing somewhere feasible: draws are
    correlated (it is MCMC), so `n_tune` sweeps are discarded as burn-in and
    `thin` can be raised if the autocorrelation is high.

    The returned `stats` record how often each kernel actually moved the chain,
    which is reported as a run diagnostic.
    """
    n_vars = x_map.shape[0]
    Z = nullspace(A_balance)
    k = Z.shape[1]

    n_skipped = 0
    n_coord_moves = n_coord_taken = 0
    n_hr_moves = n_hr_taken = 0

    def _stats():
        return {
            "n_dimensions": int(k),
            "coordinate_moves": int(n_coord_moves),
            "coordinate_move_rate": float(n_coord_taken / max(n_coord_moves, 1)),
            "hit_and_run_moves": int(n_hr_moves),
            "hit_and_run_move_rate": float(n_hr_taken / max(n_hr_moves, 1)),
            "skipped_degenerate_updates": int(n_skipped),
        }

    if k == 0:
        # Constraints pin the solution completely; nothing left to sample.
        samples = np.repeat(x_map[None, None, :], n_draws, axis=1)
        samples = np.repeat(samples, n_chains, axis=0)
        return PosteriorSamples(samples=samples, x_map=x_map,
                                method="truncated_gibbs", stats=_stats())

    Q_eff = Z.T @ Q @ Z
    Q_eff = 0.5 * (Q_eff + Q_eff.T)

    # Whitening. Coordinate Gibbs mixes at the rate of the worst conditional
    # correlation, and this posterior is strongly correlated, so sampling in raw
    # nullspace coordinates converges poorly. Reparameterize x = x_map + W u so
    # that u has precision I; the feasible set remains a polytope.
    #
    # Computed by eigendecomposition with a relative floor rather than by
    # inverting a Cholesky factor: Q_eff is severely ill-conditioned here, where
    # the latter is numerically unstable.
    w_eig, V = np.linalg.eigh(Q_eff)
    w_max = float(w_eig.max()) if w_eig.size else 1.0
    w_eig = np.maximum(w_eig, 1e-12 * max(w_max, 1.0))
    W = Z @ (V / np.sqrt(w_eig))

    # Per-coordinate RELATIVE tolerance. The entries of W span many orders of
    # magnitude, so a fixed absolute cutoff is unsafe: an entry barely above it
    # divides an O(1) numerator into an enormous step bound, which drives u to
    # an absurd value and overflows the next matrix product. An entry that is
    # negligible relative to its own column does not meaningfully constrain that
    # coordinate, so it is skipped.
    col_scale = np.abs(W).max(axis=0)
    col_tol = np.maximum(col_scale * 1e-10, np.finfo(float).tiny)

    cols = []
    for j in range(k):
        wj = W[:, j]
        t = col_tol[j]
        pos = np.where(wj > t)[0]
        neg = np.where(wj < -t)[0]
        cols.append((wj, pos, neg, wj[pos], wj[neg]))

    def _interval(pos, neg, v_pos, v_neg, base):
        """Feasible interval for the step size along a direction from `base`."""
        lo, hi = -np.inf, np.inf
        if pos.size:
            lo = max(lo, float(((x_lb[pos] - base[pos]) / v_pos).max()))
            hi = min(hi, float(((x_ub[pos] - base[pos]) / v_pos).min()))
        if neg.size:
            lo = max(lo, float(((x_ub[neg] - base[neg]) / v_neg).max()))
            hi = min(hi, float(((x_lb[neg] - base[neg]) / v_neg).min()))
        return lo, hi

    # Chains must START feasible, or the first interval is empty and the sampler
    # has nowhere valid to go. x_map can sit slightly outside the box when the
    # MAP solve stops on its iteration cap, so repair it here by alternating
    # projections in the offset coordinates dx = x - x_map, which keeps
    # A x = b exactly because A dx = 0.
    dx = np.zeros(n_vars)
    lo_d, hi_d = x_lb - x_map, x_ub - x_map
    for _ in range(500):
        if np.all(dx >= lo_d - 1e-12) and np.all(dx <= hi_d + 1e-12):
            break
        dx = np.clip(dx, lo_d, hi_d)
        dx = project_to_mass_balance(dx, A_balance)
    x_start = x_map + dx
    if not (np.all(x_start >= x_lb - 1e-8) and np.all(x_start <= x_ub + 1e-8)):
        warnings.warn(
            "truncated_gibbs_sample_nullspace: could not find a start point "
            "satisfying both the bounds and mass balance; the MAP solve is "
            "likely infeasible or the constraints are inconsistent.",
            stacklevel=2,
        )
    u_start = np.linalg.lstsq(W, x_start - x_map, rcond=None)[0]

    samples = np.empty((n_chains, n_draws, n_vars), dtype=float)

    for ci in range(n_chains):
        rng = np.random.default_rng(seed + ci)
        u = u_start.copy()
        x = x_map + W @ u
        kept = 0
        n_sweeps = n_tune + n_draws * thin

        for it in range(n_sweeps):
            for j in range(k):
                wj, pos, neg, w_pos, w_neg = cols[j]
                if pos.size == 0 and neg.size == 0:
                    # Direction unconstrained by the box: plain N(0, 1).
                    u[j] = rng.standard_normal()
                    continue
                uj = u[j]
                r = x - wj * uj      # x without this coordinate contribution
                lo, hi = _interval(pos, neg, w_pos, w_neg, r)
                if not (hi > lo) or not np.isfinite(lo) or not np.isfinite(hi):
                    n_skipped += 1
                    continue
                val = _draw_truncnorm(0.0, 1.0, lo, hi, rng)
                if not np.isfinite(val):
                    n_skipped += 1
                    continue
                n_coord_moves += 1
                if abs(val - uj) > 1e-12:
                    n_coord_taken += 1
                u[j] = val
                x = r + wj * val

            # Hit-and-run: global moves along random directions, which the
            # axis-aligned sweep above cannot make.
            for _ in range(n_direction_moves):
                d = rng.standard_normal(k)
                nrm = np.linalg.norm(d)
                if nrm < 1e-300:
                    continue
                d /= nrm
                v = W @ d
                v_tol = max(float(np.abs(v).max()) * 1e-10, np.finfo(float).tiny)
                pos = np.where(v > v_tol)[0]
                neg = np.where(v < -v_tol)[0]
                if pos.size == 0 and neg.size == 0:
                    continue
                lo, hi = _interval(pos, neg, v[pos], v[neg], x)
                if not (hi > lo) or not np.isfinite(lo) or not np.isfinite(hi):
                    n_skipped += 1
                    continue
                # Isotropic target, so along a unit direction the conditional is
                # N(-u.d, 1), truncated to the feasible segment.
                t = _draw_truncnorm(-float(u @ d), 1.0, lo, hi, rng)
                if not np.isfinite(t):
                    n_skipped += 1
                    continue
                n_hr_moves += 1
                if abs(t) > 1e-12:
                    n_hr_taken += 1
                u = u + t * d
                x = x + t * v

            # Recompute from u each sweep so mass balance stays exact rather
            # than accumulating floating-point drift across sweeps.
            x = x_map + W @ u

            if it >= n_tune and (it - n_tune) % thin == 0 and kept < n_draws:
                samples[ci, kept, :] = x
                kept += 1

    if n_skipped:
        warnings.warn(
            f"truncated_gibbs_sample_nullspace: skipped {n_skipped} degenerate "
            f"updates (empty or non-finite feasible interval). A few are normal "
            f"where bounds pin a flow; a large count means the polytope is "
            f"nearly empty.",
            stacklevel=2,
        )

    return PosteriorSamples(samples=samples, x_map=x_map,
                            method="truncated_gibbs", stats=_stats())
