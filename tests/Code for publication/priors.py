#!/usr/bin/env python3
"""
priors.py
Transfer reconciled 2017 posterior summaries into informative priors
for the 2022 update.

Reads:
  • 2017 posterior summary  (pymc_full_space_res_2017/*_posterior_mean.csv)
  • 2017 allocation posterior (if available)

Writes:
  • input data 2022/flow_prior_2022.csv
  • input data 2022/allocation_prior_2022.csv
"""

import os
import numpy as np
import pandas as pd
from io_utils import load_csv, save_csv, ensure_dir

# ── config ──────────────────────────────────────────────────────────
PRIOR_INFLATION = 1.5   # inflate 2017 posterior σ by this factor
MIN_PRIOR_STD = 0.5     # floor on prior std (kt/y)
DEFAULT_ALLOC_KAPPA = 20.0  # moderate Dirichlet concentration


def build_flow_priors(posterior_2017_path: str,
                      output_path: str,
                      inflation: float = PRIOR_INFLATION):
    """Read 2017 posterior means + HDI and produce 2022 flow priors."""
    df = load_csv(posterior_2017_path,
                  numeric_cols=["var_idx", "posterior_mean",
                                "y_median", "sigma_obs_used"])
    if "posterior_mean" not in df.columns:
        raise ValueError("posterior_mean column not found")

    df["prior_mean"] = df["posterior_mean"]

    # estimate posterior σ from HDI if available, else use obs σ
    if "hdi_2.5%" in df.columns and "hdi_97.5%" in df.columns:
        df["hdi_2.5%"] = pd.to_numeric(df["hdi_2.5%"], errors="coerce")
        df["hdi_97.5%"] = pd.to_numeric(df["hdi_97.5%"], errors="coerce")
        df["post_std"] = (df["hdi_97.5%"] - df["hdi_2.5%"]) / 3.92
    else:
        df["post_std"] = pd.to_numeric(
            df.get("sigma_obs_used", np.nan), errors="coerce")

    df["prior_std"] = np.maximum(
        df["post_std"].fillna(MIN_PRIOR_STD) * inflation,
        MIN_PRIOR_STD)

    out = df[["var_idx", "prior_mean", "prior_std"]].copy()
    if "from_node_name" in df.columns:
        out["from_node_name"] = df["from_node_name"]
    if "to_node_name" in df.columns:
        out["to_node_name"] = df["to_node_name"]

    save_csv(out, output_path)
    print(f"Flow priors: {len(out)} variables written to {output_path}")
    return out


def build_allocation_priors(posterior_2017_path: str,
                            output_path: str,
                            kappa: float = DEFAULT_ALLOC_KAPPA):
    """Build Dirichlet allocation priors from 2017 posterior flows.

    For each source node, compute the posterior-mean allocation vector
    and write it with a concentration parameter \kappa.
    """
    df = load_csv(posterior_2017_path,
                  numeric_cols=["var_idx", "from_node_number",
                                "to_node_number", "posterior_mean"])
    if "posterior_mean" not in df.columns or "from_node_number" not in df.columns:
        print("Skipping allocation priors (insufficient columns).")
        return pd.DataFrame()

    records = []
    for src, grp in df.groupby("from_node_number"):
        total = grp["posterior_mean"].sum()
        if total <= 0:
            continue
        for _, row in grp.iterrows():
            share = row["posterior_mean"] / total
            records.append(dict(
                node_id=int(src),
                target_node_id=int(row["to_node_number"]),
                var_idx=int(row["var_idx"]),
                share_mean=float(share),
                kappa=float(kappa),
            ))

    out = pd.DataFrame(records)
    save_csv(out, output_path)
    print(f"Allocation priors: {len(out)} rows written to {output_path}")
    return out


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior_2017",
                    default="pymc_full_space_res_2017/"
                            "pymc_full_space_res_2017_posterior_mean.csv")
    ap.add_argument("--output_dir", default="input data 2022")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    build_flow_priors(
        args.posterior_2017,
        os.path.join(args.output_dir, "flow_prior_2022.csv"))
    build_allocation_priors(
        args.posterior_2017,
        os.path.join(args.output_dir, "allocation_prior_2022.csv"))