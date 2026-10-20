#!/usr/bin/env python3
"""
observations.py
Map heterogeneous raw data records (flows, throughputs, allocation
shares, aggregates) to ARCADE model variables.

Produces a unified observation table consumed by the inference scripts.
"""

import numpy as np
import pandas as pd
from io_utils import (load_flows, load_scores, load_csv,
                      normalize_name, save_csv, ensure_dir)
from pedigree import score_dataframe, DIMENSIONS


VALUE_COLS = ("Value1", "Value2", "Value3", "Value4")


def pick_score_std_column(df: pd.DataFrame):
    return next((c for c in ["obs_std", "std", "sigma", "noise_std",
                             "Score", "score", "stddev"]
                 if c in df.columns), None)


def merge_scores(df_data: pd.DataFrame,
                 df_score: pd.DataFrame) -> pd.DataFrame:
    """Left-join score table onto data table."""
    if df_score.empty:
        return df_data.copy()

    join_cols = []
    if ("Flow index" in df_data.columns and
            "Flow index" in df_score.columns):
        join_cols = ["Flow index"]
    else:
        join_cols = [c for c in ["from_node_name", "to_node_name"]
                     if c in df_data.columns and c in df_score.columns]
    if not join_cols:
        return df_data.copy()

    return df_data.merge(df_score, on=join_cols, how="left",
                         suffixes=("", "_score"))


def assign_var_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a contiguous 0-based var_idx to each unique arc."""
    df = df.copy()
    if ("Flow index" in df.columns and
            df["Flow index"].notna().any()):
        zero_based = int(df["Flow index"].dropna().min()) == 0
        offset = 0 if zero_based else 1
        df["var_idx"] = df["Flow index"].apply(
            lambda v: int(v) - offset if pd.notna(v) else np.nan)
    else:
        if "from_node_name" in df.columns and "to_node_name" in df.columns:
            merge_on = ["from_node_name", "to_node_name"]
        else:
            merge_on = ["from_node_number", "to_node_number"]
        pairs = df[merge_on].drop_duplicates().reset_index(drop=True)
        pairs["var_idx"] = np.arange(len(pairs))
        df = df.merge(pairs, on=merge_on, how="left")
    df = df.loc[df["var_idx"].notna()].copy()
    df["var_idx"] = df["var_idx"].astype(int)
    return df


def extract_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Wide Value1..Value4 → long (var_idx, rep, y)."""
    present = [c for c in VALUE_COLS if c in df.columns]
    if not present:
        raise ValueError("No Value columns found")
    parts = []
    for c in present:
        y = pd.to_numeric(df[c], errors="coerce")
        tmp = pd.DataFrame({"var_idx": df["var_idx"].astype(int),
                            "rep": c, "y": y})
        parts.append(tmp.dropna(subset=["y"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["var_idx", "rep", "y"])


def build_var_map(df: pd.DataFrame) -> pd.DataFrame:
    """One row per var_idx with metadata and median observed value."""
    agg = {"from_node_number": "first", "to_node_number": "first"}
    for c in ["from_node_name", "to_node_name", "obs_std"]:
        if c in df.columns:
            agg[c] = "first" if c != "obs_std" else "median"
    vm = df.groupby("var_idx", as_index=False).agg(agg)
    return vm.sort_values("var_idx").reset_index(drop=True)


def build_observation_table(input_dir: str):
    """Main entry: load, merge, score, return (var_map, rep_long, n_vars)."""
    df_data = load_flows(input_dir)
    df_score = load_scores(input_dir)
    df = merge_scores(df_data, df_score)

    # pedigree → sigma if dimension columns present
    has_dims = all(d in df.columns for d in DIMENSIONS)
    if has_dims:
        df["y_median"] = pd.to_numeric(df.get("Value1", np.nan),
                                       errors="coerce")
        df = score_dataframe(df, value_col="y_median")
        df["obs_std"] = df["obs_sigma"]
    else:
        col = pick_score_std_column(df)
        if col:
            df["obs_std"] = pd.to_numeric(df[col], errors="coerce")

    df = assign_var_idx(df)
    var_map = build_var_map(df)
    rep_long = extract_replicates(df)
    n_vars = var_map.shape[0]

    # attach median y to var_map
    y_med = rep_long.groupby("var_idx")["y"].median()
    var_map["y_median"] = var_map["var_idx"].map(y_med).to_numpy()

    return var_map, rep_long, n_vars


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input data 2017")
    ap.add_argument("--output_dir", default="input data 2017")
    args = ap.parse_args()

    vm, rl, nv = build_observation_table(args.input_dir)
    save_csv(vm, f"{args.output_dir}/var_map.csv")
    save_csv(rl, f"{args.output_dir}/replicates_long.csv")
    print(f"Variables: {nv}, Replicates: {len(rl)}")