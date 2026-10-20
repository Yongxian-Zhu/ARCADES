"""Regression tests for the Laplace-in-nullspace sampler and the 2017->2022
prior transfer.

Both cover bugs that were silent: they produced finite, plausible-looking
numbers with the wrong uncertainty attached, and the rest of the suite passed
either way.
"""

import numpy as np

from arcade_mfa_aluminum.graph import nullspace
from arcade_mfa_aluminum.inference.sampling import laplace_sample_nullspace
from arcade_mfa_aluminum.transfer import precision_from_cov


def _random_precision(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def test_laplace_samples_match_target_covariance_unconstrained():
    """Empirical covariance of the draws must match inv(Z^T Q Z).

    Guards the Cholesky branch. The pre-fix code solved against L instead of
    L.T, yielding (L^T L)^-1: same eigenvalues, wrong eigenvectors, so this
    assertion fails on the off-diagonals while every marginal still looks
    superficially reasonable.
    """
    n = 6
    Q = _random_precision(n, seed=1)
    A = np.zeros((0, n))                     # unconstrained -> Z = I
    x_map = np.zeros(n)
    bounds = np.full(n, 1e6)

    post = laplace_sample_nullspace(
        x_map, Q, A, -bounds, bounds, n_chains=1, n_draws=200_000, seed=7
    )
    emp = np.cov(post.samples.reshape(-1, n), rowvar=False)
    target = np.linalg.inv(Q)

    # ~200k draws: Monte Carlo error on each entry is well under 2% of scale.
    assert np.allclose(emp, target, rtol=0.05, atol=0.02 * np.abs(target).max())


def test_laplace_samples_match_target_covariance_with_constraints():
    """Same check with real mass-balance rows, so Z is a non-trivial basis."""
    n, m = 7, 3
    rng = np.random.default_rng(2)
    Q = _random_precision(n, seed=2)
    A = rng.standard_normal((m, n))
    Z = nullspace(A)
    x_map = np.zeros(n)
    bounds = np.full(n, 1e6)

    post = laplace_sample_nullspace(
        x_map, Q, A, -bounds, bounds, n_chains=1, n_draws=200_000, seed=11
    )
    flat = post.samples.reshape(-1, n)

    target = Z @ np.linalg.inv(Z.T @ Q @ Z) @ Z.T
    emp = np.cov(flat, rowvar=False)
    assert np.allclose(emp, target, rtol=0.05, atol=0.02 * np.abs(target).max())


def test_laplace_samples_respect_mass_balance():
    n, m = 7, 3
    rng = np.random.default_rng(3)
    Q = _random_precision(n, seed=3)
    A = rng.standard_normal((m, n))
    bounds = np.full(n, 1e6)

    post = laplace_sample_nullspace(
        np.zeros(n), Q, A, -bounds, bounds, n_chains=2, n_draws=200, seed=5
    )
    resid = post.samples.reshape(-1, n) @ A.T
    assert np.abs(resid).max() < 1e-8


def test_precision_from_cov_is_psd_for_rank_deficient_input():
    """A covariance confined to a subspace -- exactly the 2017 transfer prior's
    shape -- must yield a valid PSD precision, not the indefinite matrix
    np.linalg.inv returns.
    """
    n, rank = 12, 7
    rng = np.random.default_rng(4)
    B = rng.standard_normal((n, rank))
    cov = B @ B.T                             # rank 7 in 12 dimensions

    P = precision_from_cov(cov)

    assert np.allclose(P, P.T)
    w = np.linalg.eigvalsh(P)
    assert w.min() > -1e-8, f"precision has negative eigenvalues: {w.min()}"
    assert np.linalg.matrix_rank(P) == rank

    # It must act as a true inverse on the supported subspace.
    P_target = np.linalg.pinv(cov)
    assert np.allclose(P, P_target, atol=1e-6 * np.abs(P_target).max())


def test_precision_from_cov_matches_inv_for_well_conditioned_input():
    cov = np.linalg.inv(_random_precision(5, seed=6))
    assert np.allclose(precision_from_cov(cov), np.linalg.inv(cov), rtol=1e-6)


def test_naive_inverse_of_singular_cov_is_indefinite():
    """Pins the reason precision_from_cov exists: document the failure mode so
    nobody 'simplifies' it back to np.linalg.inv.
    """
    n, rank = 12, 7
    rng = np.random.default_rng(4)
    B = rng.standard_normal((n, rank))
    cov = B @ B.T + 1e-8 * np.eye(n)          # the jitter build_2022_prior adds

    naive = np.linalg.inv(cov)
    assert np.linalg.eigvalsh(naive).min() < 0, (
        "expected np.linalg.inv on a jitter-regularized singular covariance to "
        "produce negative eigenvalues"
    )
    assert np.linalg.eigvalsh(precision_from_cov(cov)).min() > -1e-8


# ---------------------------------------------------------------------------
# Truncated-normal (constrained) sampler
# ---------------------------------------------------------------------------

from scipy import stats  # noqa: E402

from arcade_mfa_aluminum.inference.sampling import truncated_gibbs_sample_nullspace  # noqa: E402


def test_truncated_gibbs_never_violates_bounds():
    """The defect that motivated this sampler: clip-then-project left 19.4% of
    the real 2022 draws outside their box. Every draw here must be feasible.
    """
    n, m = 8, 3
    rng = np.random.default_rng(10)
    Q = _random_precision(n, seed=10)
    A = rng.standard_normal((m, n))
    x_map = np.zeros(n)
    # Bounds deliberately tight and asymmetric so truncation is always active.
    x_lb = np.full(n, -0.4)
    x_ub = np.full(n, 0.9)

    post = truncated_gibbs_sample_nullspace(
        x_map, Q, A, x_lb, x_ub, n_chains=2, n_draws=400, n_tune=200, seed=1
    )
    S = post.samples.reshape(-1, n)

    assert S.min() >= x_lb.min() - 1e-9, f"draw below lower bound: {S.min()}"
    assert S.max() <= x_ub.max() + 1e-9, f"draw above upper bound: {S.max()}"
    assert (S >= x_lb - 1e-9).all() and (S <= x_ub + 1e-9).all()


def test_truncated_gibbs_preserves_mass_balance_exactly():
    n, m = 8, 3
    rng = np.random.default_rng(11)
    Q = _random_precision(n, seed=11)
    A = rng.standard_normal((m, n))
    x_map = np.zeros(n)                      # A @ x_map = 0 = b
    post = truncated_gibbs_sample_nullspace(
        x_map, Q, A, np.full(n, -0.5), np.full(n, 0.5),
        n_chains=1, n_draws=300, n_tune=100, seed=2,
    )
    resid = post.samples.reshape(-1, n) @ A.T
    assert np.abs(resid).max() < 1e-9, f"mass balance drifted: {np.abs(resid).max()}"


def test_truncated_gibbs_matches_analytic_truncated_normal_1d():
    """With no constraints and one variable, the sampler must reproduce a
    known univariate truncated normal -- this is what pins it to the *correct*
    distribution, not merely to a feasible region.
    """
    sigma, lo, hi = 2.0, -1.0, 3.0
    Q = np.array([[1.0 / sigma**2]])
    post = truncated_gibbs_sample_nullspace(
        np.zeros(1), Q, np.zeros((0, 1)), np.array([lo]), np.array([hi]),
        n_chains=1, n_draws=40_000, n_tune=500, seed=3,
    )
    draws = post.samples.reshape(-1)

    ref = stats.truncnorm(lo / sigma, hi / sigma, loc=0.0, scale=sigma)
    assert abs(draws.mean() - ref.mean()) < 0.05
    assert abs(draws.std() - ref.std()) < 0.05
    # Distributional agreement, not just the first two moments.
    assert stats.kstest(draws, ref.cdf).pvalue > 1e-3


def test_truncated_gibbs_recovers_untruncated_gaussian_when_bounds_are_slack():
    """With bounds far outside the bulk, the constrained sampler must collapse
    onto the same covariance the Laplace sampler targets.
    """
    n = 5
    Q = _random_precision(n, seed=12)
    wide = np.full(n, 50.0)
    post = truncated_gibbs_sample_nullspace(
        np.zeros(n), Q, np.zeros((0, n)), -wide, wide,
        n_chains=2, n_draws=30_000, n_tune=500, seed=4,
    )
    emp = np.cov(post.samples.reshape(-1, n), rowvar=False)
    target = np.linalg.inv(Q)
    assert np.allclose(emp, target, rtol=0.15, atol=0.05 * np.abs(target).max())


def test_truncated_gibbs_respects_nonnegativity():
    """A physical MFA requirement the old sampler broke: 31 flows came back
    with negative posterior means in the 2022 run.
    """
    n, m = 6, 2
    rng = np.random.default_rng(13)
    Q = _random_precision(n, seed=13)
    A = rng.standard_normal((m, n))
    x_map = np.full(n, 1.0)
    A = A - A.mean(axis=1, keepdims=True)     # ensure A @ x_map = 0

    post = truncated_gibbs_sample_nullspace(
        x_map, Q, A, np.zeros(n), np.full(n, 10.0),
        n_chains=2, n_draws=500, n_tune=200, seed=5,
    )
    S = post.samples.reshape(-1, n)
    assert S.min() >= -1e-9, f"negative flow sampled: {S.min()}"
    assert (S.mean(axis=0) > 0).all()


def test_hit_and_run_preserves_correctness():
    """The hit-and-run kernel must leave the target invariant, not just move
    faster. Same analytic 1-D check, with direction moves switched on.
    """
    sigma, lo, hi = 2.0, -1.0, 3.0
    Q = np.array([[1.0 / sigma**2]])
    post = truncated_gibbs_sample_nullspace(
        np.zeros(1), Q, np.zeros((0, 1)), np.array([lo]), np.array([hi]),
        n_chains=1, n_draws=40_000, n_tune=500, n_direction_moves=8, seed=21,
    )
    draws = post.samples.reshape(-1)
    ref = stats.truncnorm(lo / sigma, hi / sigma, loc=0.0, scale=sigma)
    assert stats.kstest(draws, ref.cdf).pvalue > 1e-3


def test_hit_and_run_still_respects_all_constraints():
    n, m = 8, 3
    rng = np.random.default_rng(22)
    Q = _random_precision(n, seed=22)
    A = rng.standard_normal((m, n))
    x_lb, x_ub = np.full(n, -0.4), np.full(n, 0.9)
    post = truncated_gibbs_sample_nullspace(
        np.zeros(n), Q, A, x_lb, x_ub,
        n_chains=2, n_draws=400, n_tune=200, n_direction_moves=8, seed=23,
    )
    S = post.samples.reshape(-1, n)
    assert (S >= x_lb - 1e-9).all() and (S <= x_ub + 1e-9).all()
    assert np.abs(S @ A.T).max() < 1e-9


def test_hit_and_run_matches_multivariate_target_under_truncation():
    """Constrained multivariate check: compare against a long reference chain
    run with a different seed and no direction moves. Both target the same
    distribution, so their moments must agree.
    """
    n, m = 6, 2
    rng = np.random.default_rng(24)
    Q = _random_precision(n, seed=24)
    A = rng.standard_normal((m, n))
    A = A - A.mean(axis=1, keepdims=True)
    x_map = np.full(n, 1.0)
    lb, ub = np.zeros(n), np.full(n, 3.0)

    a = truncated_gibbs_sample_nullspace(
        x_map, Q, A, lb, ub, n_chains=2, n_draws=20_000, n_tune=2_000,
        n_direction_moves=8, seed=25,
    ).samples.reshape(-1, n)
    b = truncated_gibbs_sample_nullspace(
        x_map, Q, A, lb, ub, n_chains=2, n_draws=20_000, n_tune=2_000,
        n_direction_moves=0, seed=26,
    ).samples.reshape(-1, n)

    assert np.abs(a.mean(axis=0) - b.mean(axis=0)).max() < 0.05
    assert np.abs(a.std(axis=0) - b.std(axis=0)).max() < 0.05


# ---------------------------------------------------------------------------
# Ill-conditioning: the gap that let the 2017 blow-up through
# ---------------------------------------------------------------------------
#
# Every test above builds Q from well-conditioned random matrices, so the whole
# suite passed while the real 2017 problem (cond(Q_eff) ~ 7e11, plus 22 flows
# pinned to a 1e-6-wide interval) overflowed the sampler. These reproduce that
# regime in miniature.

def _ill_conditioned_precision(n, cond, seed):
    rng = np.random.default_rng(seed)
    Qr, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w = np.logspace(0, np.log10(cond), n)
    return Qr @ np.diag(w) @ Qr.T


def test_sampler_survives_extreme_conditioning():
    """cond(Q) ~ 1e12: the old absolute 1e-12 column cutoff produced bounds of
    order 1e12 here, then overflowed on the next matmul.
    """
    n, m = 10, 3
    rng = np.random.default_rng(30)
    Q = _ill_conditioned_precision(n, 1e12, seed=30)
    A = rng.standard_normal((m, n))
    x_map = np.zeros(n)
    lb, ub = np.full(n, -2.0), np.full(n, 2.0)

    post = truncated_gibbs_sample_nullspace(
        x_map, Q, A, lb, ub, n_chains=2, n_draws=300, n_tune=200,
        n_direction_moves=4, seed=31,
    )
    S = post.samples.reshape(-1, n)
    assert np.isfinite(S).all(), "sampler produced non-finite draws"
    assert (S >= lb - 1e-8).all() and (S <= ub + 1e-8).all()
    assert np.abs(S @ A.T).max() < 1e-7


def test_sampler_handles_degenerate_pinned_bounds():
    """2017 has 22 flows whose bounds are 1e-6 apart (zero-valued observations).
    A near-zero-width interval must not destabilize the chain.
    """
    n, m = 10, 2
    rng = np.random.default_rng(32)
    Q = _random_precision(n, seed=32)
    A = rng.standard_normal((m, n))
    A = A - A.mean(axis=1, keepdims=True)
    x_map = np.zeros(n)
    lb, ub = np.full(n, -1.0), np.full(n, 1.0)
    lb[:3] = -1e-6           # three flows pinned to a 1e-6 sliver
    ub[:3] = 0.0

    post = truncated_gibbs_sample_nullspace(
        x_map, Q, A, lb, ub, n_chains=2, n_draws=300, n_tune=200,
        n_direction_moves=4, seed=33,
    )
    S = post.samples.reshape(-1, n)
    assert np.isfinite(S).all()
    assert (S >= lb - 1e-9).all() and (S <= ub + 1e-9).all()
    assert np.abs(S[:, :3]).max() <= 1e-6 + 1e-9


def test_sampler_repairs_an_infeasible_x_map():
    """A MAP that stopped on its iteration cap can sit just outside the box.
    The chain must start from a repaired feasible point, not diverge.
    """
    n, m = 8, 2
    rng = np.random.default_rng(34)
    Q = _random_precision(n, seed=34)
    A = rng.standard_normal((m, n))
    A = A - A.mean(axis=1, keepdims=True)
    x_map = np.zeros(n)
    lb, ub = np.full(n, -1.0), np.full(n, 1.0)
    lb[0] = 0.05             # x_map[0] = 0 now violates its lower bound

    post = truncated_gibbs_sample_nullspace(
        x_map, Q, A, lb, ub, n_chains=1, n_draws=200, n_tune=200, seed=35
    )
    S = post.samples.reshape(-1, n)
    assert np.isfinite(S).all()
    assert (S >= lb - 1e-8).all() and (S <= ub + 1e-8).all()
