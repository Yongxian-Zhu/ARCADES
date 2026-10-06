"""
Generic Bayesian static MFA reconciliation using PyMC.
Provides full posterior distributions for all flows.
"""

import os
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pymc as pm
import arviz as az


def build_observations_log(
    df: pd.DataFrame,
    value_cols: list,
    quality_cols: list,
    quality_to_rel_sigma: dict,
    default_rel_sigma: float = 0.5,
) -> tuple:
    """Build log-space observation arrays for the likelihood."""
    rows = []
    for _, r in df.iterrows():
        fidx = int(r["flow_idx"])
        for vcol, qcol in zip(value_cols, quality_cols + [None]*len(value_cols)):
            v = r[vcol]
            if pd.notna(v) and np.isfinite(v) and v > 0:
                q = r[qcol] if qcol and qcol in r else None
                if pd.isna(q):
                    rel_sigma = default_rel_sigma
                else:
                    rel_sigma = quality_to_rel_sigma.get(
                        str(q).strip().lower(), default_rel_sigma
                    )
                sigma_log = float(np.sqrt(np.log1p(rel_sigma ** 2)))
                rows.append((fidx, float(np.log(v)), sigma_log))
    if not rows:
        return np.array([], int), np.array([]), np.array([])
    arr = np.array(rows)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]


def build_auto_balance(df: pd.DataFrame, y_median: np.ndarray,
                       sigma_rel: float = 0.001,
                       sigma_floor: float = 0.1) -> tuple:
    """Construct soft mass-balance constraints for the Bayesian model."""
    nodes_from = set(df["from_node_number"])
    nodes_to   = set(df["to_node_number"])
    internal = sorted(nodes_from & nodes_to)
    n_flows = len(df)

    y_safe = np.where(np.isfinite(y_median) & (y_median > 0), y_median, 0.0)

    rows, sigmas, nodes_used = [], [], []
    for node in internal:
        in_idx  = df.loc[df["to_node_number"]   == node, "flow_idx"].to_numpy(int)
        out_idx = df.loc[df["from_node_number"] == node, "flow_idx"].to_numpy(int)
        if not len(in_idx) or not len(out_idx) or (len(in_idx) + len(out_idx) < 3):
            continue
        row = np.zeros(n_flows)
        row[in_idx]  += 1.0
        row[out_idx] -= 1.0
        rows.append(row)
        nodes_used.append(node)
        throughput = max(y_safe[in_idx].sum(), y_safe[out_idx].sum(), 1.0)
        sigmas.append(max(sigma_rel * throughput, sigma_floor))

    A = np.vstack(rows) if rows else np.zeros((0, n_flows))
    return A, np.zeros(A.shape[0]), np.array(sigmas), nodes_used


def build_pymc_model(
    n_flows: int,
    obs_flow_idx: np.ndarray,
    obs_value_log: np.ndarray,
    obs_sigma_log: np.ndarray,
    A_all: np.ndarray,
    b_all: np.ndarray,
    s_all: np.ndarray,
    y_median: np.ndarray,
    log_prior_sigma: np.ndarray,
    upper_bounds: np.ndarray = None,
    min_positive: float = 0.1,
) -> pm.Model:
    """Construct the PyMC reconciliation model."""
    finite_y = y_median[np.isfinite(y_median) & (y_median > 0)]
    typical = float(np.median(finite_y)) if finite_y.size else 100.0

    y_safe = np.where(np.isfinite(y_median) & (y_median > 0), y_median, typical)
    y_safe = np.maximum(y_safe, min_positive)

    if upper_bounds is not None:
        ub_mask = np.isfinite(upper_bounds) & (upper_bounds > 0)
        y_safe[ub_mask] = 0.1

    coords = {"flow": np.arange(n_flows)}

    with pm.Model(coords=coords) as model:
        mu_log = np.log(y_safe)

        log_flows_raw = pm.Normal("log_flows_raw", mu=0.0, sigma=1.0,
                                  shape=n_flows, dims="flow")
        log_flows = pm.Deterministic("log_flows",
                                     mu_log + log_prior_sigma * log_flows_raw,
                                     dims="flow")
        flows = pm.Deterministic("flows", pt.exp(log_flows), dims="flow")

        if len(obs_value_log) > 0:
            pm.Normal("y_like_log",
                      mu=log_flows[obs_flow_idx],
                      sigma=obs_sigma_log,
                      observed=obs_value_log)

        if upper_bounds is not None:
            ub_idx = np.where(np.isfinite(upper_bounds) & (upper_bounds > 0))[0]
            if len(ub_idx) > 0:
                ub_vals = upper_bounds[ub_idx]
                softness = np.maximum(0.5 * ub_vals, 0.1)
                excess = pt.maximum(flows[ub_idx] - ub_vals, 0.0)
                pm.Potential("upper_bound_penalty",
                             -0.5 * pt.sum((excess / softness) ** 2))

        if A_all.shape[0] > 0:
            pm.Normal("constraints",
                      mu=pt.dot(A_all, flows),
                      sigma=s_all,
                      observed=b_all)

    return model


def reconcile_bayesian(
    df: pd.DataFrame,
    value_cols: list,
    quality_cols: list,
    config: dict,
    A_manual: np.ndarray = None,
    b_manual: np.ndarray = None,
    s_manual: np.ndarray = None,
) -> az.InferenceData:
    """Run full Bayesian reconciliation and return InferenceData."""
    n_flows = len(df)

    obs_idx, obs_v, obs_s = build_observations_log(
        df, value_cols, quality_cols,
        config.get("quality_to_sigma", {"high": 0.05, "medium": 0.15, "low": 1.0}),
        config.get("default_rel_sigma", 0.5),
    )

    y_median = np.full(n_flows, np.nan)
    for _, r in df.iterrows():
        vals = [r[c] for c in value_cols if pd.notna(r[c]) and r[c] > 0]
        if vals:
            y_median[int(r["flow_idx"])] = float(np.median(vals))

    upper_bounds = df["upper_bound"].to_numpy() if "upper_bound" in df.columns else None

    A_auto, b_auto, s_auto, _ = build_auto_balance(
        df, y_median,
        sigma_rel=config.get("balance_sigma_rel", 0.001),
        sigma_floor=config.get("balance_sigma_floor", 0.1),
    )

    if A_manual is not None and A_manual.shape[0] > 0:
        A_all = np.vstack([A_auto, A_manual])
        b_all = np.concatenate([b_auto, b_manual])
        s_all = np.concatenate([s_auto, s_manual])
    else:
        A_all, b_all, s_all = A_auto, b_auto, s_auto

    # Log-prior sigmas
    log_prior_sigma = np.full(n_flows, 1.2)
    for _, r in df.iterrows():
        fidx = int(r["flow_idx"])
        has_obs = any(pd.notna(r[c]) and r[c] > 0 for c in value_cols)
        if has_obs:
            log_prior_sigma[fidx] = 2.0     # weak prior to avoid double counting
        elif pd.notna(r.get("upper_bound", np.nan)):
            log_prior_sigma[fidx] = 0.5

    model = build_pymc_model(
        n_flows, obs_idx, obs_v, obs_s,
        A_all, b_all, s_all,
        y_median, log_prior_sigma, upper_bounds,
    )

    with model:
        idata = pm.sample(
            draws=config.get("n_draws", 2000),
            tune=config.get("n_tune", 2000),
            chains=config.get("n_chains", 4),
            target_accept=config.get("target_accept", 0.95),
            random_seed=config.get("seed", 42),
            return_inferencedata=True,
            progressbar=True,
            init="jitter+adapt_diag",
        )

    return idata