"""
arcade_mfa_aluminum.attribution
-------------------------------
What actually constrains each flow?

A reconciled flow estimate can come from very different places: a direct
observation, several disagreeing observations, the transferred prior, or purely
structural relationships such as mass balance and allocation ratios. Reporting a
credible interval without saying which of these produced it leaves readers
unable to tell an empirically grounded number from a structurally implied one.

For a Gaussian posterior the total precision decomposes additively::

    Q = Q_observations + Q_prior + Q_soft_constraints

so each flow's share of ``diag(Q)`` from each term is a direct answer to "what
is holding this flow in place". Those shares are reported alongside two
complementary measures:

* **posterior contraction** ``1 - Var_post / Var_prior`` -- how much the data
  narrowed the flow relative to the prior. Near zero means the prior is doing
  the work.
* **prior-to-posterior shift**, in prior standard deviations -- how far the data
  moved the flow. A large shift with high contraction is a flow the observations
  genuinely determined.

Precision shares describe what constrains a flow; contraction and shift describe
what the data did to it. Both are needed: a flow can be dominated by observation
precision yet barely move, if the observation happened to agree with the prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_shares(
    q_obs_diag: np.ndarray,
    Q_prior: np.ndarray | None,
    soft_blocks: dict | None = None,
) -> pd.DataFrame:
    """Per-flow share of total precision contributed by each term.

    Parameters
    ----------
    q_obs_diag : (n,) precision from the observation likelihood.
    Q_prior : (n, n) prior precision, or None when no prior is used.
    soft_blocks : mapping of label -> (n, n) precision matrix for each soft
        constraint block (e.g. mass balance, allocation ratios).

    Returns a frame with one column per term plus `total_precision`. Columns sum
    to 1 across terms for every flow with non-zero precision.
    """
    n = len(q_obs_diag)
    terms = {"observations": np.asarray(q_obs_diag, dtype=float).copy()}
    if Q_prior is not None:
        terms["prior"] = np.diag(Q_prior).astype(float).copy()
    for label, M in (soft_blocks or {}).items():
        terms[label] = np.diag(M).astype(float).copy()

    total = np.zeros(n)
    for v in terms.values():
        total += v

    out = pd.DataFrame({"flow_idx": np.arange(n), "total_precision": total})
    for label, v in terms.items():
        out[f"share_{label}"] = np.divide(
            v, total, out=np.zeros_like(v), where=total > 0
        )
    return out


def observation_inventory(obs: pd.DataFrame, n_flows: int) -> pd.DataFrame:
    """Per-flow observation counts and the sources reporting them."""
    rows = []
    grouped = obs.groupby("flow_idx") if len(obs) else {}
    for i in range(n_flows):
        if len(obs) and i in grouped.groups:
            g = grouped.get_group(i)
            rows.append({
                "flow_idx": i,
                "n_observations": len(g),
                "sources": "; ".join(sorted(g["source"].astype(str))),
                "obs_min": float(g["value"].min()),
                "obs_max": float(g["value"].max()),
                "obs_spread_pct": float(
                    100.0 * (g["value"].max() - g["value"].min())
                    / max(abs(g["value"].mean()), 1e-9)
                ) if len(g) > 1 else 0.0,
            })
        else:
            rows.append({"flow_idx": i, "n_observations": 0, "sources": "",
                         "obs_min": np.nan, "obs_max": np.nan, "obs_spread_pct": np.nan})
    return pd.DataFrame(rows)


def prior_posterior_comparison(
    samples: np.ndarray,
    prior_mean: np.ndarray | None,
    prior_cov: np.ndarray | None,
) -> pd.DataFrame:
    """Posterior contraction and prior-to-posterior shift per flow."""
    n = samples.shape[1]
    post_mean = samples.mean(axis=0)
    post_sd = samples.std(axis=0)
    out = pd.DataFrame({
        "flow_idx": np.arange(n),
        "posterior_mean": post_mean,
        "posterior_sd": post_sd,
    })
    if prior_mean is None or prior_cov is None:
        out["prior_mean"] = np.nan
        out["prior_sd"] = np.nan
        out["contraction"] = np.nan
        out["shift_in_prior_sd"] = np.nan
        return out

    prior_sd = np.sqrt(np.maximum(np.diag(prior_cov), 0.0))
    out["prior_mean"] = prior_mean
    out["prior_sd"] = prior_sd
    with np.errstate(divide="ignore", invalid="ignore"):
        out["contraction"] = np.where(
            prior_sd > 0, 1.0 - (post_sd ** 2) / (prior_sd ** 2), np.nan
        )
        out["shift_in_prior_sd"] = np.where(
            prior_sd > 0, (post_mean - prior_mean) / prior_sd, np.nan
        )
    return out


def classify_flows(
    shares: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    observation_dominant: float = 0.5,
) -> pd.DataFrame:
    """Label each flow by what principally determines it.

    - `observation-driven`  : observations supply most of the precision
    - `multi-source`        : as above, and more than one source reports it
    - `prior-informed`      : the transferred prior supplies most of it
    - `structure-determined`: mass balance and allocation ratios dominate, with
      no direct observation -- the value follows from the network, not from data
    """
    df = shares.merge(inventory, on="flow_idx", how="left")
    obs = df.get("share_observations", pd.Series(0.0, index=df.index)).fillna(0.0)
    pri = df.get("share_prior", pd.Series(0.0, index=df.index)).fillna(0.0)
    struct_cols = [c for c in df.columns
                   if c.startswith("share_") and c not in ("share_observations", "share_prior")]
    struct = df[struct_cols].sum(axis=1) if struct_cols else pd.Series(0.0, index=df.index)

    label = np.where(
        obs >= observation_dominant,
        np.where(df["n_observations"].fillna(0) > 1, "multi-source", "observation-driven"),
        np.where(pri >= struct, "prior-informed", "structure-determined"),
    )
    df["determined_by"] = label
    return df
