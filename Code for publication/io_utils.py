#!/usr/bin/env python3
"""
io_utils.py
Shared I/O helpers, name normalization, and table loaders used by every
other ARCADE module.
"""

import os
import re
import numpy as np
import pandas as pd


# ── name normalization (same logic as core script) ──────────────────
def normalize_name(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^\w_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ── generic CSV loader with optional column normalization ───────────
def load_csv(path: str, normalize_cols=None, numeric_cols=None,
             required_cols=None) -> pd.DataFrame:
    """Read a CSV, optionally normalise selected text columns and
    coerce selected columns to numeric."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, dtype=str)
    if normalize_cols:
        for c in normalize_cols:
            if c in df.columns:
                df[c] = df[c].apply(normalize_name)
    if numeric_cols:
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return df


# ── load node table ─────────────────────────────────────────────────
def load_nodes(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "nodes.csv")
    df = load_csv(path,
                  normalize_cols=["node_name"],
                  numeric_cols=["node_id"],
                  required_cols=["node_id"])
    return df.sort_values("node_id").reset_index(drop=True)


# ── load flow / arc table ──────────────────────────────────────────
def load_flows(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "flow_data.csv")
    df = load_csv(path,
                  normalize_cols=["from_node_name", "to_node_name"],
                  numeric_cols=["Flow index", "from_node_number",
                                "to_node_number",
                                "Value1", "Value2", "Value3", "Value4"])
    return df


# ── load score / pedigree table ────────────────────────────────────
def load_scores(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "flow_data_score.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = load_csv(path,
                  normalize_cols=["from_node_name", "to_node_name"],
                  numeric_cols=["Flow index", "coverage", "frequency",
                                "spatial_boundary", "obs_std", "std",
                                "sigma", "noise_std", "Score", "score",
                                "stddev"])
    return df


# ── load allocation priors ─────────────────────────────────────────
def load_allocation_priors(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "allocation_prior.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return load_csv(path, numeric_cols=["node_id", "target_node_id",
                                        "share_mean", "kappa"])


# ── load flow priors (e.g. 2017 posterior → 2022 prior) ────────────
def load_flow_priors(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "flow_prior.csv")
    if not os.path.exists(path):
        path2 = os.path.join(input_dir, "flow_prior_2022.csv")
        if not os.path.exists(path2):
            return pd.DataFrame()
        path = path2
    return load_csv(path, numeric_cols=["var_idx", "prior_mean",
                                        "prior_std"])


# ── save helpers ────────────────────────────────────────────────────
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f"  → saved {path}")