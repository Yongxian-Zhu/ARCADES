#!/usr/bin/env python3
"""
inference_map.py
Maximum-a-posteriori reconciliation via quadratic programming.

Solves:
    min_x  \Sigma_i  (x_i − μ_i)² / (2 σ_i²)
    s.t.   A_eq x = b_eq
           G x ≤ h
           lb ≤ x ≤ ub

Uses scipy.optimize.minimize (SLSQP) or, if available, cvxpy.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

from io_utils import save_csv, ensure_dir
from observations import build_observation_table
from constraints import build_all_constraints

# ── config ──────────────────────────────────────────────────────────
DEFAULT_REL_SIGMA = 0.60
DEFAULT_SIGMA_FLOOR = 1e-3


def build_qp(var_map, rep_long, n_vars, cdict):
    """Assemble QP components from observation + constraint data."""
    # target: median observed value per variable
    y_med = rep_long.groupby("var_idx")["y"].median().reindex(
        range(n_vars)).to_numpy(dtype=float)
    obs_mask = np.isfinite(y_med)

    sigma = var_map["obs_std"].to_numpy(dtype=float) if "obs_std" in var_map.columns else np.full(n_vars, np.nan)
    sigma = np.where(
        np.isfinite(sigma) & (sigma > 0), sigma,
        np.where(obs_mask,
                 np.maximum(DEFAULT_REL_SIGMA * np.abs(y_med),
                            DEFAULT_SIGMA_FLOOR),
                 1e6))

    mu = np.where(obs_mask, y_med, 0.0)
    return mu, sigma, obs_mask


def solve_map(input_dir: str, output_dir: str):
    """Run MAP reconciliation and save results."""
    var_map, rep_long, n_vars = build_observation_table(input_dir)

    from io_utils import load_flows
    df_flows = load_flows(input_dir)
    # attach var_idx to df_flows for constraint builder
    pairs = var_map[["var_idx", "from_node_number", "to_node_number"]]
    df_flows = df_flows.merge(pairs, on=["from_node_number", "to_node_number"],
                              how="left")

    cdict = build_all_constraints(input_dir, df_flows, n_vars)
    mu, sigma, obs_mask = build_qp(var_map, rep_long, n_vars, cdict)

    # objective: weighted least squares
    w = 1.0 / (sigma ** 2)

    def objective(x):
        return 0.5 * np.sum(w * (x - mu) ** 2)

    def gradient(x):
        return w * (x - mu)

    # constraints
    cons = []
    A_eq, b_eq = cdict["A_eq"], cdict["b_eq"]
    if A_eq.shape[0] > 0:
        cons.append(LinearConstraint(A_eq, b_eq - 1e-10, b_eq + 1e-10))

    G, h = cdict["G_ineq"], cdict["h_ineq"]
    if G.shape[0] > 0:
        cons.append(LinearConstraint(G, -np.inf, h))

    bounds = Bounds(cdict["lb"], cdict["ub"])

    x0 = np.clip(mu, cdict["lb"] + 1e-8, cdict["ub"] - 1e-8)
    x0 = np.where(np.isfinite(x0), x0, 1.0)

    result = minimize(objective, x0, jac=gradient, method="SLSQP",
                      bounds=bounds, constraints=cons,
                      options={"maxiter": 5000, "ftol": 1e-12,
                               "disp": True})

    print(f"MAP solver status: {result.message}")
    print(f"Objective value: {result.fun:.6g}")

    # mass-balance residual
    if A_eq.shape[0] > 0:
        resid = A_eq @ result.x - b_eq
        print(f"Mass-balance residual: max|r| = {np.max(np.abs(resid)):.2e}, "
              f"L2 = {np.linalg.norm(resid):.2e}")

    # save
    ensure_dir(output_dir)
    var_map["map_estimate"] = result.x
    var_map["y_median"] = mu
    save_csv(var_map, os.path.join(output_dir, "map_solution.csv"))
    print(f"MAP solution saved to {output_dir}/map_solution.csv")
    return result.x, var_map


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input data 2017")
    ap.add_argument("--output_dir", default="map_res_2017")
    args = ap.parse_args()
    solve_map(args.input_dir, args.output_dir)