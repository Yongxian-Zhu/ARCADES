"""
arcade_mfa_aluminum.diagnostics
-------------------------------
MCMC convergence diagnostics for the posterior draws.

Split-Rhat and ESS are what distinguish a converged chain from one that merely
ran to completion. They are computed on every run so that unreliable credible
intervals are visible rather than silent.

Both estimators follow Vehtari et al. (2021), "Rank-normalization, folding, and
localization: An improved Rhat":

* `split_rhat` halves each chain before comparing between- and within-chain
  variance, so a single chain that drifts is caught rather than averaged away.
* `effective_sample_size` uses the initial-positive-sequence truncation of the
  autocorrelation sum, computed by FFT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-Rhat per parameter.

    Parameters
    ----------
    chains : (n_chains, n_draws, n_params)

    Returns
    -------
    (n_params,) array. Values near 1.0 indicate agreement; >1.01 warrants a
    look, >1.05 means the chains have not mixed.
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws, n_params = chains.shape
    if n_draws < 4:
        return np.full(n_params, np.nan)

    half = n_draws // 2
    split = np.concatenate([chains[:, :half, :], chains[:, half:2 * half, :]], axis=0)
    m, n = split.shape[0], split.shape[1]
    if m < 2 or n < 2:
        return np.full(n_params, np.nan)

    chain_means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = n * chain_means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * within + between / n

    # A parameter pinned to a constant has zero within-chain variance and is
    # trivially converged; report 1.0 rather than dividing by zero.
    return np.sqrt(np.divide(var_hat, within, out=np.ones_like(within), where=within > 0))


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Effective sample size per parameter, via FFT autocorrelations.

    Parameters
    ----------
    chains : (n_chains, n_draws, n_params)
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws, n_params = chains.shape
    total = float(n_chains * n_draws)
    out = np.empty(n_params)

    for p in range(n_params):
        x = chains[:, :, p]
        dev = x - x.mean(axis=1, keepdims=True)
        acov = np.zeros(n_draws)
        for c in range(n_chains):
            f = np.fft.rfft(dev[c], n=2 * n_draws)
            acov += np.fft.irfft(f * np.conj(f))[:n_draws]
        if acov[0] <= 0:
            out[p] = total          # constant parameter
            continue
        rho = acov / acov[0]
        # Geyer's initial positive sequence: sum paired autocorrelations until
        # the pair goes negative, which is where the estimate becomes noise.
        s = 0.0
        for t in range(1, n_draws - 1, 2):
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            s += pair
        out[p] = total / max(1.0 + 2.0 * s, 1.0)
    return out


def convergence_summary(
    chains: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    rhat_warn: float = 1.05,
    ess_warn: float = 400.0,
) -> pd.DataFrame:
    """Per-parameter convergence table.

    Columns: flow_idx, rhat, ess, mean, sd, q2.5, q97.5, converged.
    `converged` is False where rhat exceeds `rhat_warn` or ess falls below
    `ess_warn`.
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws, n_params = chains.shape
    flat = chains.reshape(-1, n_params)

    rhat = split_rhat(chains)
    ess = effective_sample_size(chains)
    q_lo, q_hi = np.percentile(flat, [2.5, 97.5], axis=0)

    return pd.DataFrame({
        "flow_idx": np.arange(n_params) if labels is None else np.asarray(labels),
        "rhat": rhat,
        "ess": ess,
        "mean": flat.mean(axis=0),
        "sd": flat.std(axis=0),
        "q2.5": q_lo,
        "q97.5": q_hi,
        "converged": (rhat <= rhat_warn) & (ess >= ess_warn),
    })
