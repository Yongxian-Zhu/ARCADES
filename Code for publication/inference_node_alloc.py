#!/usr/bin/env python3
"""
inference_node_alloc.py
Bayesian reconciliation using the NODE–ALLOCATION parameterisation.

Latent variables
────────────────
  For each source node i with outgoing arcs {j₁, j₂, …, jₖ}:
    T_i   ~ prior          total outflow (non-negative)
    w_i   ~ Dirichlet(α)   allocation vector on the k-simplex
    x_{ij} = T_i · w_{ij}  reconstructed arc flows

Likelihoods
───────────
  • Replicate observations:  y_{f,r} ~ Normal(x_f, σ_f)
  • Allocation observations: w̃_i ~ Dirichlet(\kappa_i · w̄_i)  (optional)
  • Soft mass balance:       A x ~ Normal(0, σ_bal)

This parameterisation automatically satisfies:
  (a) \Sigma_j w_{ij} = 1   for every source node i
  (b) x_{ij} ≥ 0       because T_i ≥ 0 and w_{ij} ≥ 0

Reads:
  ./input data 2017/flow_data.csv
  ./input data 2017/flow_data_score.csv   (optional)
  ./input data 2017/allocation_prior.csv  (optional Dirichlet priors)

Outputs to:
  ./pymc_node_alloc_res_2017/
"""

import os
import re
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

import pymc as pm
import pytensor.tensor as pt

from io_utils import (load_flows, load_csv, load_allocation_priors,
                      load_flow_priors, ensure_dir, save_csv)
from observations import build_observation_table, extract_replicates
from constraints import build_mass_balance
from allocations import (build_allocation_groups, build_allocation_obs,
                          get_source_nodes_with_multiple_targets)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

INPUT_DIR = "input data 2017"
OUT_DIR = "pymc_node_alloc_res_2017"
OUT_PREFIX = "pymc_node_alloc_res_2017"

N_CHAINS = 4
N_DRAWS = 1000
N_TUNE = 1000
SEED = 42
TARGET_ACCEPT = 0.92

SIGMA_BALANCE = 1e-3          # soft mass-balance tolerance
DEFAULT_REL_SIGMA = 0.60      # fallback obs σ = 0.60 · |y|
DEFAULT_SIGMA_FLOOR = 1e-3
MIN_POSITIVE = 1e-12

# Dirichlet prior defaults
DEFAULT_KAPPA = 5.0            # weak uniform-ish Dirichlet
ALLOC_PRIOR_KAPPA = 20.0       # moderate informative Dirichlet


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _safe_dirichlet_alpha(shares: np.ndarray, kappa: float,
                          floor: float = 0.1) -> np.ndarray:
    """Compute Dirichlet α = \kappa · w, with a floor to avoid α → 0."""
    alpha = kappa * np.array(shares, dtype=float)
    alpha = np.maximum(alpha, floor)
    return alpha


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Load observations and build allocation structure ─────────
    var_map, rep_long, n_vars = build_observation_table(INPUT_DIR)
    df_flows_raw = load_flows(INPUT_DIR)

    # attach var_idx to raw flow table
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
    src_to_pos = {s: i for i, s in enumerate(source_nodes)}

    print(f"Node–allocation model: {n_sources} source nodes, "
          f"{n_vars} arc flows")

    # ── 2. Observed medians and sigma per arc ───────────────────────
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

    # ── 3. Replicate table with position mapping ────────────────────
    var_ids = var_map["var_idx"].to_numpy(dtype=int)
    pos_map_dict = {vid: pos for pos, vid in enumerate(var_ids)}
    rep_long = rep_long.copy()
    rep_long["pos"] = rep_long["var_idx"].map(pos_map_dict).astype(int)
    rep_pos = rep_long["pos"].to_numpy(dtype=int)
    rep_y = rep_long["y"].to_numpy(dtype=float)

    # ── 4. Mass-balance matrix ──────────────────────────────────────
    A_bal, _, mb_nodes = build_mass_balance(df_flows, n_vars)
    print(f"Mass-balance constraints: {A_bal.shape[0]} rows")

    # ── 5. Load optional Dirichlet allocation priors ────────────────
    alloc_prior_df = load_allocation_priors(INPUT_DIR)
    # index: source_node → {target_nodes: [...], shares: [...], kappa: float}
    alloc_prior_map = {}
    if not alloc_prior_df.empty:
        for src, grp in alloc_prior_df.groupby("node_id"):
            alloc_prior_map[int(src)] = {
                "targets": grp["target_node_id"].astype(int).tolist(),
                "shares": grp["share_mean"].astype(float).tolist(),
                "kappa": float(grp["kappa"].iloc[0]),
            }

    # ── 6. Load optional flow priors (for T_i) ─────────────────────
    flow_prior_df = load_flow_priors(INPUT_DIR)
    flow_prior_mu = {}
    flow_prior_sd = {}
    if not flow_prior_df.empty:
        for _, r in flow_prior_df.iterrows():
            idx = int(r["var_idx"])
            flow_prior_mu[idx] = float(r["prior_mean"])
            flow_prior_sd[idx] = float(r["prior_std"])

    # ── 7. Build per-source-node metadata ───────────────────────────
    # For each source node, collect:
    #   arc_indices  : list of var_idx (0-based positions in x vector)
    #   obs_total    : sum of observed medians on outgoing arcs
    #   obs_shares   : observed allocation vector
    #   n_out        : number of outgoing arcs

    node_meta = {}
    for src in source_nodes:
        grp = alloc_groups.loc[alloc_groups["source_node"] == src].copy()
        grp = grp.sort_values("pos_in_group").reset_index(drop=True)

        arc_positions = grp["var_idx"].tolist()  # these are var_idx values
        # map to 0-based positions in x vector
        arc_pos_in_x = [pos_map_dict[v] for v in arc_positions]

        obs_shares = grp["obs_share"].to_numpy(dtype=float)
        obs_total = float(grp["obs_total"].iloc[0])
        n_out = len(arc_positions)

        # Dirichlet alpha
        if src in alloc_prior_map:
            prior_info = alloc_prior_map[src]
            # align prior shares to current target ordering
            target_order = grp["target_node"].tolist()
            prior_shares = np.ones(n_out) / n_out  # fallback
            kappa = prior_info["kappa"]
            for k, tgt in enumerate(target_order):
                if tgt in prior_info["targets"]:
                    pidx = prior_info["targets"].index(tgt)
                    prior_shares[k] = prior_info["shares"][pidx]
            # renormalise
            prior_shares = prior_shares / prior_shares.sum()
            alpha = _safe_dirichlet_alpha(prior_shares, kappa)
        else:
            # use observed shares as weak Dirichlet prior
            alpha = _safe_dirichlet_alpha(
                obs_shares if np.all(np.isfinite(obs_shares)) and obs_shares.sum() > 0
                else np.ones(n_out) / n_out,
                DEFAULT_KAPPA)

        # T_i prior: use sum of observed outgoing flows
        if np.isfinite(obs_total) and obs_total > 0:
            T_mu = obs_total
            T_sd = max(DEFAULT_REL_SIGMA * obs_total, DEFAULT_SIGMA_FLOOR)
        else:
            # check flow priors
            prior_vals = [flow_prior_mu.get(v, np.nan) for v in arc_positions]
            if any(np.isfinite(v) for v in prior_vals):
                T_mu = float(np.nansum(prior_vals))
                T_sd = max(0.5 * T_mu, DEFAULT_SIGMA_FLOOR)
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

    # ── 8. Identify "orphan" arcs not covered by any source node ────
    # (e.g. boundary inflows with no outgoing arcs from their source)
    covered_positions = set()
    for meta in node_meta.values():
        covered_positions.update(meta["arc_pos_in_x"])
    orphan_positions = sorted(set(range(n_vars)) - covered_positions)
    print(f"Arcs covered by allocation groups: {len(covered_positions)}")
    print(f"Orphan arcs (modelled as independent flows): {len(orphan_positions)}")

    # ── 9. PyMC model ───────────────────────────────────────────────
    typical_scale = float(np.nanmedian(np.abs(y_median[obs_mask]))) \
        if np.any(obs_mask) else 1.0
    typical_scale = max(typical_scale, 1.0)

    with pm.Model() as model:

        # --- T_i and w_i for each source node ---
        T_vars = {}    # source_node → PyMC variable
        w_vars = {}    # source_node → PyMC variable (Dirichlet)
        x_parts = {}   # position_in_x → PyTensor expression

        for src in source_nodes:
            meta = node_meta[src]
            n_out = meta["n_out"]

            # ── total throughput T_i ──
            if meta["T_mu"] is not None:
                T_i = pm.TruncatedNormal(
                    f"T[{src}]",
                    mu=meta["T_mu"],
                    sigma=meta["T_sd"],
                    lower=MIN_POSITIVE,
                )
            else:
                T_i = pm.HalfNormal(f"T[{src}]", sigma=typical_scale)
            T_vars[src] = T_i

            # ── allocation vector w_i ──
            if n_out == 1:
                # trivial: single outgoing arc, w = [1.0]
                w_i = pt.as_tensor_variable([1.0])
                # no need to register as a random variable
            else:
                w_i = pm.Dirichlet(f"w[{src}]", a=meta["alpha"])
            w_vars[src] = w_i

            # ── reconstruct arc flows: x_{ij} = T_i * w_{ij} ──
            for k, pos in enumerate(meta["arc_pos_in_x"]):
                if n_out == 1:
                    x_parts[pos] = T_i  # w = 1
                else:
                    x_parts[pos] = T_i * w_i[k]

        # --- orphan arcs (not part of any allocation group) ---
        for pos in orphan_positions:
            if obs_mask[pos] and np.isfinite(sigma_obs[pos]):
                mu0 = float(y_median[pos])
                sd0 = float(max(sigma_obs[pos], DEFAULT_SIGMA_FLOOR))
                xj = pm.TruncatedNormal(
                    f"x_orphan[{pos}]", mu=mu0, sigma=sd0,
                    lower=MIN_POSITIVE)
            else:
                xj = pm.HalfNormal(f"x_orphan[{pos}]",
                                   sigma=typical_scale)
            x_parts[pos] = xj

        # --- assemble full x vector ---
        x_list = [x_parts[pos] for pos in range(n_vars)]
        x = pt.stack(x_list)
        pm.Deterministic("x", x)

        # --- also register T and w as deterministics for output ---
        T_arr = pt.stack([T_vars[s] for s in source_nodes])
        pm.Deterministic("T", T_arr)

        for src in source_nodes:
            meta = node_meta[src]
            if meta["n_out"] > 1:
                pm.Deterministic(f"w_det[{src}]", w_vars[src])

        # ── replicate likelihood ──
        pm.Normal(
            "y_like",
            mu=x[rep_pos],
            sigma=sigma_obs[rep_pos],
            observed=rep_y,
        )

        # ── optional: Dirichlet likelihood on observed allocations ──
        # (strengthens allocation signal when direct share data exist)
        for src in source_nodes:
            meta = node_meta[src]
            if meta["n_out"] < 2:
                continue
            obs_s = meta["obs_shares"]
            if not (np.all(np.isfinite(obs_s)) and np.all(obs_s > 0)):
                continue
            # observed allocation as Dirichlet likelihood
            obs_alpha = _safe_dirichlet_alpha(obs_s, ALLOC_PRIOR_KAPPA)
            pm.Dirichlet(
                f"w_obs_like[{src}]",
                a=obs_alpha,
                observed=w_vars[src],
            )

        # ── soft mass balance ──
        if A_bal.shape[0] > 0:
            Ax = pt.dot(pt.as_tensor_variable(A_bal), x)
            pm.Normal("mass_balance", mu=0.0, sigma=SIGMA_BALANCE,
                      observed=Ax)

        # ── sample ──
        trace = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=min(4, N_CHAINS),
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            progressbar=True,
        )

    idata = az.from_pymc(trace, model=model)

    # ── 10. Save outputs ────────────────────────────────────────────
    ensure_dir(OUT_DIR)

    # ArviZ summary for x
    summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
    save_csv(summ.reset_index(),
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_pymc_summary.csv"))

    # posterior means
    post_x = idata.posterior["x"]
    x_mean = post_x.mean(dim=("chain", "draw")).to_numpy()

    pm_df = var_map.copy()
    pm_df["y_median"] = y_median
    pm_df["sigma_obs_used"] = sigma_obs
    pm_df["posterior_mean"] = x_mean
    save_csv(pm_df,
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_posterior_mean.csv"))

    # T posterior means
    post_T = idata.posterior["T"]
    T_mean = post_T.mean(dim=("chain", "draw")).to_numpy()
    T_df = pd.DataFrame({
        "source_node": source_nodes,
        "T_posterior_mean": T_mean,
    })
    save_csv(T_df,
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_T_posterior_mean.csv"))

    # w posterior means (per source node with ≥ 2 targets)
    w_records = []
    for src in source_nodes:
        meta = node_meta[src]
        if meta["n_out"] < 2:
            # single target → w = 1
            w_records.append(dict(
                source_node=src,
                target_node=meta["arc_positions"][0],
                var_idx=meta["arc_positions"][0],
                w_posterior_mean=1.0,
            ))
            continue
        vname = f"w_det[{src}]"
        if vname in idata.posterior:
            w_post = idata.posterior[vname].mean(
                dim=("chain", "draw")).to_numpy()
        else:
            # fallback: compute from x
            arc_means = x_mean[meta["arc_pos_in_x"]]
            total = arc_means.sum()
            w_post = arc_means / total if total > 0 else np.ones(
                meta["n_out"]) / meta["n_out"]

        grp = alloc_groups.loc[
            alloc_groups["source_node"] == src
        ].sort_values("pos_in_group").reset_index(drop=True)

        for k in range(meta["n_out"]):
            w_records.append(dict(
                source_node=src,
                target_node=int(grp.iloc[k]["target_node"]),
                var_idx=int(grp.iloc[k]["var_idx"]),
                w_posterior_mean=float(w_post[k]),
                w_prior_alpha=float(meta["alpha"][k]),
            ))

    w_df = pd.DataFrame(w_records)
    save_csv(w_df,
             os.path.join(OUT_DIR, f"{OUT_PREFIX}_w_posterior_mean.csv"))

    # mass-balance residual
    if A_bal.shape[0] > 0:
        resid = A_bal @ x_mean
        print(f"Mass-balance residual (posterior mean): "
              f"max|r| = {np.max(np.abs(resid)):.2e}, "
              f"L2 = {np.linalg.norm(resid):.2e}")

    # NetCDF trace
    az.to_netcdf(idata,
                 os.path.join(OUT_DIR, f"{OUT_PREFIX}.nc"))

    # ── 11. Posterior histograms for T and selected w ────────────────
    # T histograms
    post_T_flat = post_T.stack(sample=("chain", "draw")).transpose(
        "T_dim_0", "sample").to_numpy()
    for i, src in enumerate(source_nodes):
        vals = post_T_flat[i]
        plt.figure(figsize=(6, 3.5))
        plt.hist(vals, bins=50, density=True, alpha=0.6, color="teal")
        mu = float(np.mean(vals))
        lo, hi = np.quantile(vals, [0.025, 0.975])
        plt.axvline(mu, color="k", ls="--", label=f"mean={mu:.3g}")
        plt.axvline(lo, color="red", ls=":", label=f"2.5%={lo:.3g}")
        plt.axvline(hi, color="red", ls=":", label=f"97.5%={hi:.3g}")
        plt.title(f"T[{src}] — total outflow from node {src}")
        plt.xlabel("T (kt/y)")
        plt.ylabel("Density")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"hist_T_{src}.png"), dpi=150)
        plt.close()

    # w histograms (selected multi-target nodes)
    multi_src = get_source_nodes_with_multiple_targets(alloc_groups)
    for src in multi_src[:20]:  # limit to first 20 for speed
        vname = f"w_det[{src}]"
        if vname not in idata.posterior:
            continue
        w_post = idata.posterior[vname]
        w_flat = w_post.stack(sample=("chain", "draw")).to_numpy()
        # w_flat shape: (n_targets, n_samples)
        meta = node_meta[src]
        grp = alloc_groups.loc[
            alloc_groups["source_node"] == src
        ].sort_values("pos_in_group").reset_index(drop=True)
        n_out = meta["n_out"]

        fig, axes = plt.subplots(1, n_out, figsize=(4 * n_out, 3.5),
                                  squeeze=False)
        for k in range(n_out):
            ax = axes[0, k]
            vals = w_flat[k]
            ax.hist(vals, bins=50, density=True, alpha=0.6,
                    color="steelblue")
            mu = float(np.mean(vals))
            ax.axvline(mu, color="k", ls="--")
            tgt = int(grp.iloc[k]["target_node"])
            ax.set_title(f"w[{src}→{tgt}]\nmean={mu:.3f}", fontsize=9)
            ax.set_xlabel("share")
            ax.set_xlim(0, 1)
        plt.suptitle(f"Allocation from node {src}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"hist_w_{src}.png"), dpi=150)
        plt.close()

    print(f"\nNode–allocation inference complete. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()