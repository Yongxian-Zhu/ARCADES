#!/usr/bin/env python3
"""
run_2022_update_mcmc.py
Bayesian update of the U.S. aluminum system from 2017 → 2022.

Reads:
  • input data 2022/flow_data.csv          (2022 observations)
  • input data 2022/flow_data_score.csv    (optional pedigree)
  • input data 2022/flow_prior_2022.csv    (2017 posterior → prior)
  • input data 2022/allocation_prior_2022.csv

Outputs to:
  • pymc_full_space_res_2022/
"""

import os
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import pytensor.tensor as pt

from io_utils import (load_csv, load_flow_priors, load_allocation_priors,
                      ensure_dir, save_csv)
from observations import build_observation_table, extract_replicates
from constraints import build_all_constraints, build_mass_balance
from io_utils import load_flows

# ── config ──────────────────────────────────────────────────────────
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

INPUT_DIR = "input data 2022"
OUT_DIR = "pymc_full_space_res_2022"
OUT_PREFIX = "pymc_full_space_res_2022"

N_CHAINS = 4
N_DRAWS = 1000
N_TUNE = 1000
SEED = 2022
TARGET_ACCEPT = 0.9
SIGMA_BALANCE = 1e-3
MIN_POSITIVE = 1e-12
DEFAULT_REL_SIGMA = 0.60
DEFAULT_SIGMA_FLOOR = 1e-3
USE_HARD_BOUNDS = True
HARD_BOUNDS_REL = (0.2, 1.8)


def main():
    # ── load 2022 observations ──────────────────────────────────────
    var_map, rep_long, n_vars = build_observation_table(INPUT_DIR)
    df_flows = load_flows(INPUT_DIR)
    pairs = var_map[["var_idx", "from_node_number", "to_node_number"]]
    df_flows = df_flows.merge(pairs,
                              on=["from_node_number", "to_node_number"],
                              how="left")

    # ── load 2017→2022 priors ───────────────────────────────────────
    flow_priors = load_flow_priors(INPUT_DIR)
    alloc_priors = load_allocation_priors(INPUT_DIR)

    # build prior arrays (indexed by var position 0..n_vars-1)
    prior_mu = np.full(n_vars, np.nan)
    prior_sd = np.full(n_vars, np.nan)
    if not flow_priors.empty:
        for _, r in flow_priors.iterrows():
            idx = int(r["var_idx"])
            if 0 <= idx < n_vars:
                prior_mu[idx] = float(r["prior_mean"])
                prior_sd[idx] = float(r["prior_std"])

    # ── observation medians and sigma ───────────────────────────────
    y_median = (rep_long.groupby("var_idx")["y"].median()
                .reindex(range(n_vars)).to_numpy(dtype=float))
    obs_mask = np.isfinite(y_median)

    sigma_obs = var_map["obs_std"].to_numpy(dtype=float) if "obs_std" in var_map.columns else np.full(n_vars, np.nan)
    sigma_obs = np.where(
        np.isfinite(sigma_obs) & (sigma_obs > 0), sigma_obs,
        np.where(obs_mask,
                 np.maximum(DEFAULT_REL_SIGMA * np.abs(y_median),
                            DEFAULT_SIGMA_FLOOR),
                 np.inf))

    # ── bounds ──────────────────────────────────────────────────────
    if USE_HARD_BOUNDS:
        max_obs = np.nanmax(y_median) if np.any(obs_mask) else 1.0
        wide_ub = max(1.0, 10.0 * max_obs)
        x_lb = np.where(obs_mask, HARD_BOUNDS_REL[0] * y_median, 0.0)
        x_ub = np.where(obs_mask, HARD_BOUNDS_REL[1] * y_median, wide_ub)
        x_lb = np.minimum(x_lb, x_ub - 1e-12)
    else:
        x_lb = np.zeros(n_vars)
        x_ub = np.full(n_vars, np.inf)

    # ── mass balance ────────────────────────────────────────────────
    A_bal, _, mb_nodes = build_mass_balance(df_flows, n_vars)

    # ── replicate index mapping ─────────────────────────────────────
    var_ids = var_map["var_idx"].to_numpy(dtype=int)
    pos_map = {vid: pos for pos, vid in enumerate(var_ids)}
    rep_long = rep_long.copy()
    rep_long["pos"] = rep_long["var_idx"].map(pos_map).astype(int)
    rep_pos = rep_long["pos"].to_numpy(dtype=int)
    rep_y = rep_long["y"].to_numpy(dtype=float)

    print(f"2022 update: {n_vars} variables, "
          f"{len(rep_y)} replicate obs, "
          f"{A_bal.shape[0]} mass-balance rows, "
          f"{(~np.isnan(prior_mu)).sum()} informative priors from 2017")

    # ── PyMC model ──────────────────────────────────────────────────
    coords = {"flow": np.arange(n_vars),
              "mb": np.arange(A_bal.shape[0])}

    with pm.Model(coords=coords) as model:
        typical_scale = float(np.nanmedian(
            np.abs(y_median[obs_mask]))) if np.any(obs_mask) else 1.0
        typical_scale = max(typical_scale, 1.0)

        x_list = []
        for j in range(n_vars):
            has_prior = np.isfinite(prior_mu[j]) and np.isfinite(prior_sd[j])
            has_obs = obs_mask[j] and np.isfinite(sigma_obs[j])

            if has_obs:
                mu0 = float(y_median[j])
                sd0 = float(max(sigma_obs[j], DEFAULT_SIGMA_FLOOR))
            elif has_prior:
                mu0 = float(prior_mu[j])
                sd0 = float(max(prior_sd[j], DEFAULT_SIGMA_FLOOR))
            else:
                mu0 = None
                sd0 = None

            if mu0 is not None:
                # combine prior and obs into a single TruncatedNormal
                # (if both exist, use obs centre; prior enters via
                #  the likelihood of the 2017 posterior mean)
                lower = float(max(x_lb[j], MIN_POSITIVE)
                              ) if USE_HARD_BOUNDS else MIN_POSITIVE
                upper = float(x_ub[j]) if (
                    USE_HARD_BOUNDS and np.isfinite(x_ub[j])) else np.inf
                xj = pm.TruncatedNormal(f"x[{j}]", mu=mu0,
                                        sigma=sd0,
                                        lower=lower, upper=upper)
            else:
                xj = pm.HalfNormal(f"x[{j}]", sigma=typical_scale)

            # additional prior likelihood from 2017 posterior
            if has_prior and has_obs:
                pm.Normal(f"prior_like_{j}", mu=prior_mu[j],
                          sigma=prior_sd[j], observed=xj)

            x_list.append(xj)

        x = pt.stack(x_list)
        pm.Deterministic("x", x, dims=("flow",))

        # replicate likelihood
        pm.Normal("y_like", mu=x[rep_pos], sigma=sigma_obs[rep_pos],
                  observed=rep_y)

        # soft mass balance
        if A_bal.shape[0] > 0:
            Ax = pt.dot(pt.as_tensor_variable(A_bal), x)
            pm.Normal("mass_balance", mu=0.0, sigma=SIGMA_BALANCE,
                      observed=Ax, dims=("mb",))

        trace = pm.sample(draws=N_DRAWS, tune=N_TUNE,
                          chains=N_CHAINS,
                          cores=min(4, N_CHAINS),
                          target_accept=TARGET_ACCEPT,
                          random_seed=SEED, progressbar=True)

    idata = az.from_pymc(trace, model=model)

    # ── save ────────────────────────────────────────────────────────
    ensure_dir(OUT_DIR)
    summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
    save_csv(summ.reset_index(),
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_pymc_summary.csv"))

    x_mean = idata.posterior["x"].mean(
        dim=("chain", "draw")).to_numpy()
    pm_df = var_map.copy()
    pm_df["posterior_mean"] = x_mean
    pm_df["y_median"] = y_median
    pm_df["sigma_obs_used"] = sigma_obs
    save_csv(pm_df,
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_posterior_mean.csv"))

    az.to_netcdf(idata,
                 os.path.join(OUT_DIR, f"{OUT_PREFIX}.nc"))

    if A_bal.shape[0] > 0:
        resid = A_bal @ x_mean
        print(f"Mass-balance residual (posterior mean): "
              f"max|r| = {np.max(np.abs(resid)):.2e}, "
              f"L2 = {np.linalg.norm(resid):.2e}")

    print(f"2022 update complete. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()