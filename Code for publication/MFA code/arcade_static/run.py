"""
Driver script for ARCADE static MFA reconciliation.
Supports both optimization and Bayesian solvers via config selection.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from arcade_static.io import (
    load_config, load_flows, compute_weighted_observation,
    parse_constraint_formula,
)
from arcade_static.optimizer import reconcile_optimization, validate_mass_balance
from arcade_static.bayesian import reconcile_bayesian


def build_manual_constraints(df_flows, constraints_path):
    """Build manual constraint matrix from CSV of formulas."""
    if not Path(constraints_path).exists():
        return None, None, None, []

    df_c = pd.read_csv(constraints_path)
    valid_nodes = set(df_flows["from_node_number"]).union(df_flows["to_node_number"])
    flow_lookup = {
        (int(r["from_node_number"]), int(r["to_node_number"])): int(r["flow_idx"])
        for _, r in df_flows.iterrows()
    }

    A_rows, b_vals, s_vals, descs = [], [], [], []
    n_flows = len(df_flows)

    for _, row in df_c.iterrows():
        terms, target = parse_constraint_formula(
            row.get("formula"), row.get("dest_node"), valid_nodes
        )
        if terms is None:
            continue
        a = np.zeros(n_flows)
        ok = True
        for coef, src, dst in terms:
            if dst is None:
                # aggregate term: include all outflows from src
                idx = df_flows.loc[df_flows["from_node_number"] == src, "flow_idx"].to_numpy(int)
                if not len(idx):
                    ok = False
                    break
                for fi in idx:
                    a[fi] += coef
            else:
                key = (src, dst)
                if key not in flow_lookup:
                    ok = False
                    break
                a[flow_lookup[key]] += coef
        if not ok:
            continue
        A_rows.append(a)
        b_vals.append(float(target))
        s_vals.append(max(0.10 * abs(target), 100.0))   # generic sigma policy
        descs.append(str(row.get("description", "")))

    if not A_rows:
        return None, None, None, []
    return np.vstack(A_rows), np.array(b_vals), np.array(s_vals), descs


def main(config_path: str):
    cfg = load_config(config_path)
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df_flows, value_cols, quality_cols = load_flows(cfg["input_files"]["flows"])
    means, uncerts = compute_weighted_observation(
        df_flows, value_cols, quality_cols,
        cfg.get("quality_to_sigma", {"high": 0.05, "medium": 0.15, "low": 1.0}),
        cfg.get("default_rel_sigma", 0.5),
    )

    constraints_path = cfg["input_files"].get("constraints", "")
    A_man, b_man, s_man, _ = build_manual_constraints(df_flows, constraints_path)

    solver = cfg.get("solver", "optimization").lower()

    if solver == "optimization":
        print("Running optimization solver…")
        result = reconcile_optimization(df_flows, means, uncerts,
                                        A_manual=A_man, b_manual=b_man)
        df_flows["reconciled_value"] = result["x"]
        df_flows.to_csv(out_dir / "reconciled_flows.csv", index=False)
        mb = validate_mass_balance(df_flows, result["x"])
        mb.to_csv(out_dir / "mass_balance.csv", index=False)
        print(f"Done. Runtime: {result['runtime']:.2f}s, "
              f"max |imbalance|%: {mb['imbalance_pct'].max():.4e}")

    elif solver == "bayesian":
        print("Running Bayesian sampler…")
        idata = reconcile_bayesian(df_flows, value_cols, quality_cols, cfg,
                                    A_manual=A_man, b_manual=b_man, s_manual=s_man)
        post = idata.posterior["flows"]
        df_flows["posterior_mean"] = post.mean(dim=("chain", "draw")).to_numpy()
        df_flows["posterior_sd"]   = post.std(dim=("chain", "draw")).to_numpy()
        df_flows["q2.5"]  = post.quantile(0.025, dim=("chain", "draw")).to_numpy()
        df_flows["q97.5"] = post.quantile(0.975, dim=("chain", "draw")).to_numpy()
        df_flows.to_csv(out_dir / "reconciled_flows.csv", index=False)
        idata.to_netcdf(out_dir / "posterior.nc")
        print("Done. Posterior saved.")

    else:
        raise ValueError(f"Unknown solver '{solver}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)