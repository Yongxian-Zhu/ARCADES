#!/usr/bin/env python3
"""
inference_node_alloc_2022.py
2022 Bayesian update using the node–allocation parameterisation.

Uses 2017 posterior T and w as informative priors, then updates with
2022 observations.

Reads:
  ./input data 2022/flow_data.csv
  ./input data 2022/flow_data_score.csv
  ./input data 2022/flow_prior_2022.csv
  ./input data 2022/allocation_prior_2022.csv

Outputs to:
  ./pymc_node_alloc_res_2022/
"""

import os
import numpy as np
import pandas as pd
import arviz as az

import pymc as pm
import pytensor.tensor as pt

from io_utils import (load_flows, load_flow_priors, load_allocation_priors,
                      ensure_dir, save_csv)
from observations import build_observation_table
from constraints import build_mass_balance
from allocations import (build_allocation_groups, build_allocation_obs,
                          get_source_nodes_with_multiple_targets)

# ── config ──────────────────────────────────────────────────────────
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

INPUT_DIR = "input data 2022"
OUT_DIR = "pymc_node_alloc_res_2022"
OUT_PREFIX = "pymc_node_alloc_res_2022"

N_CHAINS = 4
N_DRAWS = 1000
N_TUNE = 1000
SEED = 2022
TARGET_ACCEPT = 0.92
SIGMA_BALANCE = 1e-3
DEFAULT_REL_SIGMA = 0.60
DEFAULT_SIGMA_FLOOR = 1e-3
MIN_POSITIVE = 1e-12
DEFAULT_KAPPA = 5.0
ALLOC_PRIOR_KAPPA = 20.0


def _safe_dirichlet_alpha(shares, kappa, floor=0.1):
    alpha = kappa * np.array(shares, dtype=float)
    return np.maximum(alpha, floor)


def main():
    # ── load 2022 data ──────────────────────────────────────────────
    var_map, rep_long, n_vars = build_observation_table(INPUT_DIR)
    df_flows_raw = load_flows(INPUT_DIR)
    pairs = var_map[["var_idx", "from_node_number", "to_node_number"]]
    df_flows = df_flows_raw.merge(
        pairs, on=["from_node_number", "to_node_number"], how="left")
    df_flows = df_flows.dropna(subset=["var_idx"])
    df_flows["var_idx"] = df_flows["var_idx"].astype(int)

    # allocation groups
    alloc_groups = build_allocation_groups(df_flows)
    alloc_groups = build_allocation_obs(alloc_groups, var_map, rep_long)
    source_nodes = sorted(alloc_groups["source_node"].unique())
    n_sources = len(source_nodes)

    # ── load 2017 → 2022 priors ────────────────────────────────────
    flow_prior_df = load_flow_priors(INPUT_DIR)
    alloc_prior_df = load_allocation_priors(INPUT_DIR)

    # flow priors → T priors (sum per source node)
    flow_prior_mu = {}
    flow_prior_sd = {}
    if not flow_prior_df.empty:
        for _, r in flow_prior_df.iterrows():
            idx = int(r["var_idx"])
            flow_prior_mu[idx] = float(r["prior_mean"])
            flow_prior_sd[idx] = float(r["prior_std"])

    # allocation priors → Dirichlet α per source node
    alloc_prior_map = {}
    if not alloc_prior_df.empty:
        for src, grp in alloc_prior_df.groupby("node_id"):
            alloc_prior_map[int(src)] = {
                "targets": grp["target_node_id"].astype(int).tolist(),
                "shares": grp["share_mean"].astype(float).tolist(),
                "kappa": float(grp["kappa"].iloc[0]),
            }

    # ── observation arrays ──────────────────────────────────────────
    y_median = (rep_long.groupby("var_idx")["y"].median()
                .reindex(range(n_vars)).to_numpy(dtype=float))
    obs_mask = np.isfinite(y_median)

    sigma_obs = (var_map["obs_std"].to_numpy(dtype=float)
                 if "obs_std" in var_map.columns
                 else np.full(n_vars, np.nan))
    sigma_obs = np.where(
        np.isfinite(sigma_obs) & (sigma_obs > 0), sigma_obs,
        np.where(obs_mask,
                 np.maximum(DEFAULT_REL_SIGMA * np.abs(y_median),
                            DEFAULT_SIGMA_FLOOR),
                 np.inf))

    var_ids = var_map["var_idx"].to_numpy(dtype=int)
    pos_map_dict = {vid: pos for pos, vid in enumerate(var_ids)}
    rep_long = rep_long.copy()
    rep_long["pos"] = rep_long["var_idx"].map(pos_map_dict).astype(int)
    rep_pos = rep_long["pos"].to_numpy(dtype=int)
    rep_y = rep_long["y"].to_numpy(dtype=float)

    A_bal, _, mb_nodes = build_mass_balance(df_flows, n_vars)

    # ── per-node metadata ───────────────────────────────────────────
    node_meta = {}
    for src in source_nodes:
        grp = alloc_groups.loc[
            alloc_groups["source_node"] == src
        ].sort_values("pos_in_group").reset_index(drop=True)

        arc_positions = grp["var_idx"].tolist()
        arc_pos_in_x = [pos_map_dict[v] for v in arc_positions]
        obs_shares = grp["obs_share"].to_numpy(dtype=float)
        obs_total = float(grp["obs_total"].iloc[0])
        n_out = len(arc_positions)

        # Dirichlet alpha (prefer 2017 prior if available)
        if src in alloc_prior_map:
            pi = alloc_prior_map[src]
            target_order = grp["target_node"].tolist()
            prior_shares = np.ones(n_out) / n_out
            kappa = pi["kappa"]
            for k, tgt in enumerate(target_order):
                if tgt in pi["targets"]:
                    pidx = pi["targets"].index(tgt)
                    prior_shares[k] = pi["shares"][pidx]
            prior_shares /= prior_shares.sum()
            alpha = _safe_dirichlet_alpha(prior_shares, kappa)
        else:
            alpha = _safe_dirichlet_alpha(
                obs_shares if np.all(np.isfinite(obs_shares))
                and obs_shares.sum() > 0
                else np.ones(n_out) / n_out,
                DEFAULT_KAPPA)

        # T prior from 2017 flow priors
        prior_vals = [flow_prior_mu.get(v, np.nan) for v in arc_positions]
        prior_sds = [flow_prior_sd.get(v, np.nan) for v in arc_positions]
        if any(np.isfinite(v) for v in prior_vals):
            T_mu = float(np.nansum(prior_vals))
            T_sd = float(np.sqrt(np.nansum(
                [s**2 for s in prior_sds if np.isfinite(s)])))
            T_sd = max(T_sd, DEFAULT_SIGMA_FLOOR)
        elif np.isfinite(obs_total) and obs_total > 0:
            T_mu = obs_total
            T_sd = max(DEFAULT_REL_SIGMA * obs_total, DEFAULT_SIGMA_FLOOR)
        else:
            T_mu = None
            T_sd = None

        node_meta[src] = dict(
            arc_positions=arc_positions,
            arc_pos_in_x=arc_pos_in_x,
            n_out=n_out,
            obs_shares=obs_shares,
            obs_total=obs_total,
            alpha=alpha,
            T_mu=T_mu,
            T_sd=T_sd,
        )

    covered = set()
    for m in node_meta.values():
        covered.update(m["arc_pos_in_x"])
    orphan_positions = sorted(set(range(n_vars)) - covered)

    typical_scale = float(np.nanmedian(np.abs(y_median[obs_mask]))) \
        if np.any(obs_mask) else 1.0
    typical_scale = max(typical_scale, 1.0)

    print(f"2022 node–alloc update: {n_sources} source nodes, "
          f"{n_vars} arcs, {len(orphan_positions)} orphans, "
          f"{A_bal.shape[0]} MB rows")

    # ── PyMC model ──────────────────────────────────────────────────
    with pm.Model() as model:
        T_vars, w_vars, x_parts = {}, {}, {}

        for src in source_nodes:
            meta = node_meta[src]
            n_out = meta["n_out"]

            if meta["T_mu"] is not None:
                T_i = pm.TruncatedNormal(
                    f"T[{src}]", mu=meta["T_mu"], sigma=meta["T_sd"],
                    lower=MIN_POSITIVE)
            else:
                T_i = pm.HalfNormal(f"T[{src}]", sigma=typical_scale)
            T_vars[src] = T_i

            if n_out == 1:
                w_i = pt.as_tensor_variable([1.0])
            else:
                w_i = pm.Dirichlet(f"w[{src}]", a=meta["alpha"])
            w_vars[src] = w_i

            for k, pos in enumerate(meta["arc_pos_in_x"]):
                x_parts[pos] = T_i * w_i[k] if n_out > 1 else T_i

        for pos in orphan_positions:
            if obs_mask[pos] and np.isfinite(sigma_obs[pos]):
                xj = pm.TruncatedNormal(
                    f"x_orphan[{pos}]",
                    mu=float(y_median[pos]),
                    sigma=float(max(sigma_obs[pos], DEFAULT_SIGMA_FLOOR)),
                    lower=MIN_POSITIVE)
            else:
                xj = pm.HalfNormal(f"x_orphan[{pos}]",
                                   sigma=typical_scale)
            x_parts[pos] = xj

        x = pt.stack([x_parts[p] for p in range(n_vars)])
        pm.Deterministic("x", x)

        # replicate likelihood
        pm.Normal("y_like", mu=x[rep_pos], sigma=sigma_obs[rep_pos],
                  observed=rep_y)

        # Dirichlet likelihood on observed allocations
        for src in source_nodes:
            meta = node_meta[src]
            if meta["n_out"] < 2:
                continue
            obs_s = meta["obs_shares"]
            if np.all(np.isfinite(obs_s)) and np.all(obs_s > 0):
                obs_alpha = _safe_dirichlet_alpha(obs_s, ALLOC_PRIOR_KAPPA)
                pm.Dirichlet(f"w_obs_like[{src}]", a=obs_alpha,
                             observed=w_vars[src])

        # soft mass balance
        if A_bal.shape[0] > 0:
            Ax = pt.dot(pt.as_tensor_variable(A_bal), x)
            pm.Normal("mass_balance", mu=0.0, sigma=SIGMA_BALANCE,
                      observed=Ax)

        trace = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                          cores=min(4, N_CHAINS),
                          target_accept=TARGET_ACCEPT,
                          random_seed=SEED, progressbar=True)

    idata = az.from_pymc(trace, model=model)

    # ── save ────────────────────────────────────────────────────────
    ensure_dir(OUT_DIR)

    summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
    save_csv(summ.reset_index(),
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_pymc_summary.csv"))

    x_mean = idata.posterior["x"].mean(dim=("chain", "draw")).to_numpy()
    pm_df = var_map.copy()
    pm_df["posterior_mean"] = x_mean
    pm_df["y_median"] = y_median
    save_csv(pm_df,
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_posterior_mean.csv"))

    az.to_netcdf(idata, os.path.join(OUT_DIR, f"{OUT_PREFIX}.nc"))

    if A_bal.shape[0] > 0:
        resid = A_bal @ x_mean
        print(f"MB residual: max|r|={np.max(np.abs(resid)):.2e}, "
              f"L2={np.linalg.norm(resid):.2e}")

    print(f"2022 node–alloc update complete. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()