"""
arcade_mfa_aluminum.transfer
--------------------
2017 -> 2022 posterior-to-prior transfer utilities.

Derives the 2022 prior mean and covariance from the 2017 posterior draws,
with an optional inflation factor so that the transferred prior is not
overconfident: the 2017 posterior is five years stale by the time it informs
2022.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arcade_mfa_aluminum.paths import prepare_output


@dataclass
class TransferPrior:
    mean: np.ndarray          # (n_flows,)
    cov: np.ndarray           # (n_flows, n_flows)
    source_year: int
    target_year: int
    inflation_factor: float


def summarize_2017_posterior(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (n_chains, n_draws, n_flows) draws into a mean and covariance."""
    flat = samples.reshape(-1, samples.shape[-1])   # (n_chains*n_draws, n_flows)
    mean = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False)
    return mean, cov


def build_2022_prior_from_2017(
    samples_2017: np.ndarray,
    *,
    inflation_factor: float = 1.5,
    jitter: float = 1e-8,
) -> TransferPrior:
    """Build the 2022 prior from 2017 posterior draws.

    `inflation_factor` multiplies the 2017 posterior covariance before it is
    used as a 2022 prior, to reflect added uncertainty from 5 years of
    structural/technological change that the 2017 data cannot speak to. A
    factor of 1.0 corresponds to no inflation; 1.5 is a conservative default.
    This is a modelling assumption rather than a fitted value, and is exposed
    as a config knob (`output.transfer_inflation_factor`).
    """
    mean, cov = summarize_2017_posterior(samples_2017)
    cov_inflated = cov * inflation_factor + jitter * np.eye(cov.shape[0])
    return TransferPrior(
        mean=mean, cov=cov_inflated,
        source_year=2017, target_year=2022,
        inflation_factor=inflation_factor,
    )


def precision_from_cov(cov: np.ndarray, *, rcond: float = 1e-10) -> np.ndarray:
    """Numerically safe precision (inverse covariance) for a possibly
    rank-deficient covariance matrix.

    The source-year posterior draws lie on the mass-balance nullspace, so their
    sample covariance is exactly singular in the constrained directions. A
    direct `np.linalg.inv` there does not merely lose accuracy: it returns a
    matrix that is not a valid precision at all (indefinite, with large
    negative eigenvalues), which makes the MAP objective non-convex.

    Instead, symmetrize, eigendecompose, and invert only those directions whose
    eigenvalue exceeds ``rcond * max_eigenvalue``. The remainder receive zero
    precision, which is the correct reading: they are the mass-balance
    subspace, already pinned by hard equality constraints in the solver, so
    nothing is gained by constraining them twice.

    Returns a symmetric positive-semidefinite matrix.
    """
    C = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    w, V = np.linalg.eigh(C)
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0.0:
        return np.zeros_like(C)
    thresh = rcond * w_max
    w_inv = np.where(w > thresh, 1.0 / np.where(w > thresh, w, 1.0), 0.0)
    P = (V * w_inv) @ V.T
    return 0.5 * (P + P.T)


def save_prior_npy(prior: TransferPrior, mean_path: str, cov_path: str) -> None:
    """Write the prior mean and covariance to .npy files."""
    np.save(prepare_output(mean_path), prior.mean)
    np.save(prepare_output(cov_path), prior.cov)
