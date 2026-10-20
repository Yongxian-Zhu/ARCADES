"""
arcade_mfa_aluminum.inference.optimization
----------------------------------
Constrained quadratic MAP solver.

Objective (quadratic, in linear flow space):

    0.5 * (x - mu)^T Sigma_prior^-1 (x - mu)   [optional prior term]
  + 0.5 * sum_j (x_j - y_j)^2 / sigma_obs_j^2   [observation term]

subject to:
    A_balance @ x == b_balance   (mass balance, equality)
    x_lb <= x <= x_ub            (bounds)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass
class MapResult:
    x_map: np.ndarray
    success: bool
    message: str
    n_iter: int
    Q: np.ndarray       # effective precision (Hessian) used -- reused for sampling
    q: np.ndarray
    # Precision contributed by each term, kept so that a flow's posterior can be
    # attributed to what actually constrains it (see arcade_mfa_aluminum.attribution).
    q_obs_diag: np.ndarray = None        # (n,) from the observation likelihood
    Q_prior: np.ndarray = None           # (n, n) from the transferred prior, or None
    Q_soft: np.ndarray = None            # (n, n) from all soft constraint rows, or None


def solve_map(
    y_obs: np.ndarray,
    sigma_obs: np.ndarray,
    A_balance: np.ndarray,
    b_balance: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    *,
    prior_mean: Optional[np.ndarray] = None,
    inv_prior_cov: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 20000,
    method: str = "trust-constr",
    soft_A: Optional[np.ndarray] = None,
    soft_b: Optional[np.ndarray] = None,
    soft_sigma: Optional[np.ndarray] = None,
    obs_idx: Optional[np.ndarray] = None,
    obs_value: Optional[np.ndarray] = None,
    obs_sigma: Optional[np.ndarray] = None,
    verbose: int = 0,
) -> MapResult:
    """Solve the constrained quadratic MAP problem.

    With `prior_mean`/`inv_prior_cov` omitted (the 2017 baseline year), the
    objective reduces to a weighted least-squares fit to the observations. When
    supplied (the 2022 update), the prior is the 2017 posterior carried forward
    by `arcade_mfa_aluminum.transfer`.

    `obs_idx`/`obs_value`/`obs_sigma` supply observations in long form, one
    entry per (flow, source). A flow observed by several sources contributes one
    likelihood term per observation, so conflicting reports are reconciled by
    their relative precision rather than by discarding all but one. When these
    are omitted the dense `y_obs`/`sigma_obs` pair is used instead, which is the
    special case of exactly one observation per flow.

    `soft_A`/`soft_b`/`soft_sigma` add an optional RELAXED linear constraint
    block, penalized rather than enforced::

        + 0.5 * (soft_A x - soft_b)^T W (soft_A x - soft_b),  W = diag(1/soft_sigma^2)

    which contributes `soft_A^T W soft_A` to Q and `-soft_A^T W soft_b` to q.
    This is how transferred allocation ratios enter the 2022 solve (see
    `arcade_mfa_aluminum.allocation`): mass balance and bounds stay hard, while
    the ratios act as an adjustable preference. Since the added term is PSD, Q
    remains PSD, and since `Q` is returned on the result and reused as the
    Laplace/Gibbs precision, the extra information propagates into the
    posterior automatically.
    """
    n = y_obs.shape[0]

    if obs_idx is not None:
        # Long form: accumulate one term per observation. A flow with several
        # observations gets the sum of their precisions, and its linear term is
        # the precision-weighted sum of the reported values -- i.e. the
        # posterior is pulled toward the inverse-variance weighted mean of the
        # sources. With one observation per flow this reduces exactly to the
        # dense branch below.
        if obs_value is None or obs_sigma is None:
            raise ValueError("obs_idx supplied without obs_value/obs_sigma")
        obs_idx = np.asarray(obs_idx, dtype=int)
        if obs_idx.size and (obs_idx.min() < 0 or obs_idx.max() >= n):
            raise ValueError(
                f"obs_idx out of range [0, {n - 1}]: "
                f"[{obs_idx.min()}, {obs_idx.max()}]"
            )
        w = 1.0 / np.maximum(np.asarray(obs_sigma, dtype=float) ** 2, 1e-12)
        inv_obs_var = np.zeros(n)
        obs_lin = np.zeros(n)
        np.add.at(inv_obs_var, obs_idx, w)
        np.add.at(obs_lin, obs_idx, w * np.asarray(obs_value, dtype=float))
    else:
        obs_mask = np.isfinite(y_obs)
        inv_obs_var = np.where(
            obs_mask, 1.0 / np.maximum(sigma_obs ** 2, 1e-12), 0.0
        )
        obs_lin = inv_obs_var * np.nan_to_num(y_obs, nan=0.0)

    q_obs_diag = inv_obs_var.copy()
    Q_prior_term = None
    Q_soft_term = None

    if inv_prior_cov is not None:
        Q_prior_term = np.asarray(inv_prior_cov, dtype=float)
        Q = inv_prior_cov + np.diag(inv_obs_var)
        mu = prior_mean if prior_mean is not None else np.zeros(n)
        q = -inv_prior_cov @ mu - obs_lin
    else:
        Q = np.diag(inv_obs_var)
        q = -obs_lin

    if soft_A is not None and soft_A.size and soft_A.shape[0] > 0:
        if soft_sigma is None:
            raise ValueError("soft_A supplied without soft_sigma")
        if soft_A.shape[1] != n:
            raise ValueError(
                f"soft_A has {soft_A.shape[1]} columns but there are {n} flows"
            )
        if soft_sigma.shape[0] != soft_A.shape[0]:
            raise ValueError(
                f"soft_sigma has {soft_sigma.shape[0]} entries for "
                f"{soft_A.shape[0]} soft-constraint rows"
            )
        w = 1.0 / np.maximum(soft_sigma ** 2, 1e-12)
        Q_soft_term = (soft_A.T * w) @ soft_A
        Q = Q + Q_soft_term
        if soft_b is not None and np.any(soft_b):
            q = q - (soft_A.T * w) @ soft_b
        Q = 0.5 * (Q + Q.T)

    def obj_fun(x):
        return 0.5 * float(x @ (Q @ x)) + float(q @ x)

    def obj_grad(x):
        return Q @ x + q

    def obj_hess(x):
        # The objective is exactly quadratic, so the Hessian is the constant Q.
        # Supplying it matters: without it trust-constr falls back to a BFGS
        # quasi-Newton approximation and rebuilds curvature that is already
        # known in closed form, which prevents convergence within any
        # reasonable iteration budget.
        return Q

    if x0 is None:
        x0 = prior_mean.copy() if prior_mean is not None else np.nan_to_num(y_obs, nan=0.0)
        if A_balance.shape[0] > 0:
            rhs = A_balance @ x0
            AAT = A_balance @ A_balance.T
            try:
                lam = np.linalg.solve(AAT + 1e-10 * np.eye(AAT.shape[0]), rhs)
            except np.linalg.LinAlgError:
                lam = np.linalg.lstsq(AAT + 1e-10 * np.eye(AAT.shape[0]), rhs, rcond=None)[0]
            x0 = x0 - A_balance.T @ lam
        x0 = np.clip(x0, x_lb, x_ub)

    lin_con = (
        LinearConstraint(A_balance, lb=b_balance, ub=b_balance)
        if A_balance.shape[0] > 0 else None
    )
    bounds = Bounds(x_lb, x_ub)

    res = minimize(
        fun=obj_fun,
        x0=x0,
        jac=obj_grad,
        hess=obj_hess,
        method=method,
        bounds=bounds,
        constraints=([lin_con] if lin_con is not None else []),
        options={"maxiter": maxiter, "verbose": verbose},
    )

    return MapResult(
        x_map=res.x,
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(getattr(res, "niter", -1)),
        Q=Q,
        q=q,
        q_obs_diag=q_obs_diag,
        Q_prior=Q_prior_term,
        Q_soft=Q_soft_term,
    )
