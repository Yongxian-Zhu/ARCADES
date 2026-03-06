#!/usr/bin/env python3
"""
allocations.py
Build the node–allocation structure from the flow graph.

For every source node i that has outgoing arcs, define:
  • T_i        = total outflow from node i
  • w_{ij}     = allocation share from node i to each downstream node j
  • x_{ij}     = T_i * w_{ij}   (reconstructed arc flow)

Outputs a table of allocation groups consumed by the inference scripts.
"""

import os
import numpy as np
import pandas as pd
from io_utils import load_csv, load_flows, save_csv, ensure_dir


def build_allocation_groups(df_flows: pd.DataFrame) -> pd.DataFrame:
    """Identify allocation groups from the flow table.

    For each source node, list all outgoing arcs with their var_idx.

    Returns
    -------
    DataFrame with columns:
        source_node, target_node, var_idx, group_id, n_targets
    where group_id == source_node (one group per source).
    """
    required = ["from_node_number", "to_node_number", "var_idx"]
    for c in required:
        if c not in df_flows.columns:
            raise ValueError(f"Missing column: {c}")

    df = df_flows[required].drop_duplicates().copy()
    df = df.dropna(subset=required)
    for c in required:
        df[c] = df[c].astype(int)

    # count targets per source
    counts = df.groupby("from_node_number")["to_node_number"].transform("count")
    df["n_targets"] = counts.astype(int)

    df = df.rename(columns={"from_node_number": "source_node",
                             "to_node_number": "target_node"})
    df["group_id"] = df["source_node"]

    # sort for deterministic ordering within each group
    df = df.sort_values(["source_node", "target_node"]).reset_index(drop=True)

    # within-group position (0-based)
    df["pos_in_group"] = df.groupby("source_node").cumcount()

    return df


def build_allocation_obs(alloc_groups: pd.DataFrame,
                         var_map: pd.DataFrame,
                         rep_long: pd.DataFrame) -> pd.DataFrame:
    """Compute observed allocation shares from replicate data.

    For each source node, compute the median observed flow on each
    outgoing arc and normalise to get empirical shares.

    Returns
    -------
    DataFrame with columns:
        source_node, target_node, var_idx, obs_share, obs_total
    """
    y_med = rep_long.groupby("var_idx")["y"].median()

    ag = alloc_groups.copy()
    ag["y_median"] = ag["var_idx"].map(y_med).astype(float)

    # total observed outflow per source
    totals = ag.groupby("source_node")["y_median"].transform("sum")
    ag["obs_total"] = totals
    ag["obs_share"] = np.where(totals > 0,
                                ag["y_median"] / totals,
                                1.0 / ag["n_targets"])  # uniform fallback
    return ag


def get_source_nodes_with_multiple_targets(alloc_groups: pd.DataFrame):
    """Return list of source nodes that have ≥ 2 outgoing arcs
    (i.e. where allocation is non-trivial)."""
    return sorted(
        alloc_groups.loc[alloc_groups["n_targets"] >= 2, "source_node"].unique()
    )


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from observations import build_observation_table

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input data 2017")
    ap.add_argument("--output_dir", default="input data 2017")
    args = ap.parse_args()

    var_map, rep_long, n_vars = build_observation_table(args.input_dir)
    df_flows = load_flows(args.input_dir)
    # attach var_idx
    pairs = var_map[["var_idx", "from_node_number", "to_node_number"]]
    df_flows = df_flows.merge(pairs,
                              on=["from_node_number", "to_node_number"],
                              how="left")

    ag = build_allocation_groups(df_flows)
    ag = build_allocation_obs(ag, var_map, rep_long)
    save_csv(ag, os.path.join(args.output_dir, "allocation_groups.csv"))

    multi = get_source_nodes_with_multiple_targets(ag)
    print(f"Allocation groups: {ag['source_node'].nunique()} source nodes, "
          f"{len(multi)} with ≥ 2 targets, {len(ag)} total arcs")