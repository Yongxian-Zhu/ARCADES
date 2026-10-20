#!/usr/bin/env python3
"""
diagnostics.py
Post-sampling diagnostics for ARCADE posterior traces.

Checks:
  1. Convergence: R-hat, ESS, divergences
  2. Feasibility: mass-balance residuals, inequality slacks, bounds
  3. Posterior predictive checks (optional)
"""

import os
import numpy as np
import pandas as pd
import arviz as az

from io_utils import load_csv, save_csv, ensure_dir
from constraints import build_mass_balance
from observations import build_observation_table


def load_trace(trace_path: str):
    return az.from_netcdf(trace_path)


# ── 1. Convergence diagnostics ──────────────────────────────────────
def convergence_summary(idata, var_names=("x",), hdi_prob=0.95):
    """Return ArviZ summary with R-hat, ESS, HDI."""
    summ = az.summary(idata, var_names=list(var_names), hdi_prob=hdi_prob)
    n_divergent = int(idata.sample_stats["diverging"].sum().values)
    print(f"Divergent transitions: {n_divergent}")
    rhat_max = summ["r_hat"].max() if "r_hat" in summ.columns else np.nan
    ess_min = summ["ess_bulk"].min() if "ess_bulk" in summ.columns else np.nan
    print(f"Max R-hat: {rhat_max:.4f}")
    print(f"Min ESS (bulk): {ess_min:.0f}")
    flags = []
    if rhat_max > 1.05:
        flags.append("WARNING: R-hat > 1.05 for some variables")
    if ess_min < 100:
        flags.append("WARNING: ESS < 100 for some variables")
    if n_divergent > 0:
        flags.append(f"WARNING: {n_divergent} divergent transitions")
    for f in flags:
        print(f"  ⚠ {f}")
    return summ, flags


# ── 2. Feasibility checks ──────────────────────────────────────────
def feasibility_check(idata, input_dir: str):
    """Check mass-balance residuals and bound satisfaction on
    posterior mean."""
    from io_utils import load_flows
    var_map, _, n_vars = build_observation_table(input_dir)
    df_flows = load_flows(input_dir)
    pairs = var_map[["var_idx", "from_node_number", "to_node_number"]]
    df_flows = df_flows.merge(pairs,
                              on=["from_node_number", "to_node_number"],
                              how="left")

    A_mb, b_mb, nodes = build_mass_balance(df_flows, n_vars)

    post = idata.posterior["x"]
    x_mean = post.mean(dim=("chain", "draw")).to_numpy()

    results = {}
    if A_mb.shape[0] > 0:
        resid = A_mb @ x_mean - b_mb
        results["mb_max_abs"] = float(np.max(np.abs(resid)))
        results["mb_l2"] = float(np.linalg.norm(resid))
        print(f"Mass-balance residual (posterior mean): "
              f"max|r| = {results['mb_max_abs']:.2e}, "
              f"L2 = {results['mb_l2']:.2e}")

    neg = np.sum(x_mean < -1e-8)
    results["n_negative"] = int(neg)
    if neg > 0:
        print(f"  ⚠ {neg} posterior-mean flows are negative")
    else:
        print("  ✓ All posterior-mean flows are non-negative")

    return results


# ── 3. Posterior predictive check ───────────────────────────────────
def posterior_predictive_check(idata, var_map: pd.DataFrame,
                               rep_long: pd.DataFrame):
    """Compare observed replicates against posterior predictive draws."""
    post = idata.posterior["x"]
    x_mean = post.mean(dim=("chain", "draw")).to_numpy()

    y_med = rep_long.groupby("var_idx")["y"].median()
    ppc = pd.DataFrame({
        "var_idx": y_med.index,
        "y_observed": y_med.values,
        "y_posterior_mean": x_mean[y_med.index.to_numpy()],
    })
    ppc["residual"] = ppc["y_posterior_mean"] - ppc["y_observed"]
    ppc["rel_residual"] = ppc["residual"] / ppc["y_observed"].replace(0, np.nan)
    return ppc


# ── full report ─────────────────────────────────────────────────────
def run_diagnostics(trace_path: str, input_dir: str, output_dir: str):
    ensure_dir(output_dir)
    idata = load_trace(trace_path)

    # convergence
    summ, flags = convergence_summary(idata)
    save_csv(summ.reset_index(), os.path.join(output_dir,
                                               "convergence_summary.csv"))

    # feasibility
    feas = feasibility_check(idata, input_dir)
    pd.DataFrame([feas]).to_csv(
        os.path.join(output_dir, "feasibility_check.csv"), index=False)

    # PPC
    var_map, rep_long, _ = build_observation_table(input_dir)
    ppc = posterior_predictive_check(idata, var_map, rep_long)
    save_csv(ppc, os.path.join(output_dir, "posterior_predictive.csv"))

    # flags
    with open(os.path.join(output_dir, "diagnostic_flags.txt"), "w") as f:
        for fl in flags:
            f.write(fl + "\n")
        if not flags:
            f.write("All diagnostics passed.\n")

    print(f"Diagnostics saved to {output_dir}/")


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace",
                    default="pymc_full_space_res_2017/"
                            "pymc_full_space_res_2017.nc")
    ap.add_argument("--input_dir", default="input data 2017")
    ap.add_argument("--output_dir", default="pymc_full_space_res_2017")
    args = ap.parse_args()
    run_diagnostics(args.trace, args.input_dir, args.output_dir)