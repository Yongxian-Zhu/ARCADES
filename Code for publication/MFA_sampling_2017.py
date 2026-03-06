#!/usr/bin/env python3
"""
Bayesian (PyMC) flow reconciliation with soft mass-balance constraints,
using the "new" long-table format where each flow can have multiple values
(Value1..Value4) with blanks meaning missing.

Reads inputs from:
    ./input data 2017/flow_data.csv
    ./input data 2017/flow_data_score.csv   

Outputs to:
    ./Outputs/

Modeling choices (kept close to your earlier logic):
  - For each flow j, latent true flow x_j > 0
  - Observed replicates y_{j,r} ~ Normal(x_j, sigma_j), for all non-missing values
  - sigma_j is taken from score file if available; else heuristic
  - Mass balance is enforced softly: A x ~ Normal(0, SIGMA_BALANCE)
  - Optional hard bounds: x_j constrained to [0.2*median(y_j), 1.8*median(y_j)] if any observed
"""

import os
import re
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

import pymc as pm
import pytensor.tensor as pt

# -------------------------
# Config
# -------------------------
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

INPUT_DIR = "input data 2017"
FLOW_FILE = os.path.join(INPUT_DIR, "flow_data.csv")
SCORE_FILE = os.path.join(INPUT_DIR, "flow_data_score.csv")  # optional

out_prefix = "pymc_full_space_res_2017"
out_dir = out_prefix

# MCMC settings
N_CHAINS = 4
N_DRAWS = 1000
N_TUNE = 1000
SEED = 42

# Mass-balance softness (smaller => stricter). Units are flow units.
SIGMA_BALANCE = 1e-3

# Optional hard bounds around observation median
USE_HARD_BOUNDS = True
HARD_BOUNDS_REL = (0.2, 1.8)
MIN_POSITIVE = 1e-12

# If score file doesn't provide obs_std, use heuristic per-flow sigma
DEFAULT_REL_SIGMA = 0.60      # sigma = 0.60*|median(y)|
DEFAULT_SIGMA_FLOOR = 1e-3


# -------------------------
# Helpers
# -------------------------
def normalize_name(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^\w_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def pick_score_std_column(df: pd.DataFrame):
    return next((c for c in ["obs_std", "std", "sigma", "noise_std", "Score", "score", "stddev"] if c in df.columns), None)


def build_mass_balance(df_raw: pd.DataFrame, var_idx_map: pd.DataFrame, n_vars: int):
    """Build A_bal x = 0 for internal nodes (appear in both from_node_number and to_node_number)."""
    from_all = df_raw["from_node_number"].dropna().astype(int).to_numpy() if "from_node_number" in df_raw.columns else np.array([], int)
    to_all   = df_raw["to_node_number"].dropna().astype(int).to_numpy()   if "to_node_number" in df_raw.columns else np.array([], int)
    internal_nodes = set(from_all).intersection(set(to_all))

    df_sub = var_idx_map.copy()
    df_sub["from_node_number"] = pd.to_numeric(df_sub.get("from_node_number", np.nan), errors="coerce")
    df_sub["to_node_number"]   = pd.to_numeric(df_sub.get("to_node_number",   np.nan), errors="coerce")

    A_rows, nodes_used = [], []
    for node in sorted(internal_nodes):
        inflow_vars  = df_sub.loc[df_sub["to_node_number"]   == node, "var_idx"].to_numpy(dtype=int)
        outflow_vars = df_sub.loc[df_sub["from_node_number"] == node, "var_idx"].to_numpy(dtype=int)
        if inflow_vars.size > 0 and outflow_vars.size > 0:
            row = np.zeros(n_vars, dtype=float)
            row[inflow_vars] += 1.0
            row[outflow_vars] -= 1.0
            A_rows.append(row)
            nodes_used.append(int(node))

    A_bal = np.vstack(A_rows) if A_rows else np.zeros((0, n_vars), dtype=float)
    return A_bal, np.zeros(A_bal.shape[0], dtype=float), nodes_used


def extract_replicates(df: pd.DataFrame, value_cols=("Value1", "Value2", "Value3", "Value4")) -> pd.DataFrame:
    """
    Convert wide replicates (Value1..Value4) into long replicate table:
      var_idx, rep, y
    Missing replicate cells are dropped.
    """
    present = [c for c in value_cols if c in df.columns]
    if not present:
        raise ValueError(f"No replicate value columns found. Expected one of: {value_cols}")

    recs = []
    for c in present:
        y = pd.to_numeric(df[c], errors="coerce")
        tmp = pd.DataFrame({"var_idx": df["var_idx"].astype(int), "rep": c, "y": y})
        tmp = tmp.loc[tmp["y"].notna()].copy()
        recs.append(tmp)

    if not recs:
        return pd.DataFrame(columns=["var_idx", "rep", "y"])

    out = pd.concat(recs, ignore_index=True)
    return out


# -------------------------
# 1) Load and preprocess
# -------------------------
if not os.path.exists(FLOW_FILE):
    raise FileNotFoundError(f"Missing input flow file: {FLOW_FILE}")

df_data = pd.read_csv(FLOW_FILE, dtype=str)
df_score = pd.read_csv(SCORE_FILE, dtype=str) if os.path.exists(SCORE_FILE) else pd.DataFrame()

# Normalize names if present
for c in ["from_node_name", "to_node_name"]:
    if c in df_data.columns:
        df_data[c] = df_data[c].apply(normalize_name)
    if c in df_score.columns:
        df_score[c] = df_score[c].apply(normalize_name)

# Numeric conversions
for c in ["Flow index", "from_node_number", "to_node_number"]:
    if c in df_data.columns:
        df_data[c] = pd.to_numeric(df_data[c], errors="coerce")

# Replicate value columns -> numeric
for c in ["Value1", "Value2", "Value3", "Value4"]:
    if c in df_data.columns:
        df_data[c] = pd.to_numeric(df_data[c], errors="coerce")

if not df_score.empty and "Flow index" in df_score.columns:
    df_score["Flow index"] = pd.to_numeric(df_score["Flow index"], errors="coerce")

# Merge score -> data
join_cols = []
if not df_score.empty and ("Flow index" in df_data.columns) and ("Flow index" in df_score.columns):
    join_cols = ["Flow index"]
else:
    join_cols = [c for c in ["from_node_name", "to_node_name"] if (c in df_data.columns and c in df_score.columns)]

if join_cols and not df_score.empty:
    df = df_data.merge(df_score, on=join_cols, how="left", suffixes=("", "_score"))
else:
    df = df_data.copy()

score_std_col = pick_score_std_column(df)
df["obs_std"] = pd.to_numeric(df[score_std_col], errors="coerce") if score_std_col else np.nan

# Need node numbers for mass balance
need_cols = ["from_node_number", "to_node_number"]
for c in need_cols:
    if c not in df.columns:
        raise ValueError(f"flow_data.csv missing required column: {c}")

df = df.loc[df["from_node_number"].notna() & df["to_node_number"].notna()].copy()

# Assign var_idx
if "Flow index" in df.columns and df["Flow index"].notna().any():
    zero_based = int(df["Flow index"].dropna().min()) == 0
    offset = 0 if zero_based else 1
    df["var_idx"] = df["Flow index"].apply(lambda v: int(v) - offset if pd.notna(v) else np.nan)
else:
    # fallback on unique (from,to) pairs
    if ("from_node_name" in df.columns) and ("to_node_name" in df.columns):
        pairs = df[["from_node_name", "to_node_name"]].drop_duplicates().reset_index(drop=True)
        merge_on = ["from_node_name", "to_node_name"]
    else:
        pairs = df[["from_node_number", "to_node_number"]].drop_duplicates().reset_index(drop=True)
        merge_on = ["from_node_number", "to_node_number"]
    pairs["var_idx"] = np.arange(len(pairs))
    df = df.merge(pairs, on=merge_on, how="left")

df = df.loc[df["var_idx"].notna()].copy()
df["var_idx"] = df["var_idx"].astype(int)

# Aggregate metadata per var_idx (keep replicates separate for likelihood)
agg_dict = dict(
    from_node_number=("from_node_number", "first"),
    to_node_number=("to_node_number", "first"),
    obs_std=("obs_std", "median"),
)
if "from_node_name" in df.columns:
    agg_dict["from_node_name"] = ("from_node_name", "first")
if "to_node_name" in df.columns:
    agg_dict["to_node_name"] = ("to_node_name", "first")

var_map = df.groupby("var_idx", as_index=False).agg(**agg_dict)
var_map = var_map.sort_values("var_idx").reset_index(drop=True)
n_vars = var_map.shape[0]
print(f"Variables detected: {n_vars}")

# Build replicate observation table
rep_long = extract_replicates(df, value_cols=("Value1", "Value2", "Value3", "Value4"))
if rep_long.empty:
    raise RuntimeError("No observed values found in Value1..Value4. Nothing to fit.")

# For each var_idx, compute a representative central value (median of available replicates)
y_median = rep_long.groupby("var_idx")["y"].median().reindex(var_map["var_idx"]).to_numpy(dtype=float)
obs_mask = np.isfinite(y_median)

# Per-variable observation sigma
sigma_obs = var_map["obs_std"].to_numpy(dtype=float)
sigma_obs = np.where(
    np.isfinite(sigma_obs),
    sigma_obs,
    np.where(obs_mask, np.maximum(DEFAULT_REL_SIGMA * np.abs(y_median), DEFAULT_SIGMA_FLOOR), np.inf),
)

# Optional bounds based on median observation
if USE_HARD_BOUNDS:
    max_obs = np.nanmax(y_median) if np.any(obs_mask) else 1.0
    wide_ub = max(1.0, 10.0 * max_obs)

    x_lb = np.where(obs_mask, HARD_BOUNDS_REL[0] * y_median, 0.0)
    x_ub = np.where(obs_mask, HARD_BOUNDS_REL[1] * y_median, wide_ub)
    x_lb = np.minimum(x_lb, x_ub - 1e-12)
else:
    x_lb = np.zeros(n_vars)
    x_ub = np.full(n_vars, np.inf)

# Align replicate table to 0..n_vars-1 indexing used by pt.stack
# Important: var_idx might not be contiguous if Flow index has gaps.
# We create a mapping from actual var_idx values to row positions in var_map.
var_ids = var_map["var_idx"].to_numpy(dtype=int)
pos_map = {vid: pos for pos, vid in enumerate(var_ids)}
rep_long["pos"] = rep_long["var_idx"].map(pos_map).astype(int)

rep_pos = rep_long["pos"].to_numpy(dtype=int)
rep_y = rep_long["y"].to_numpy(dtype=float)

# -------------------------
# 2) Mass-balance constraints
# -------------------------
A_bal, _, mb_nodes = build_mass_balance(df_data, var_map.assign(var_idx=np.arange(n_vars)), n_vars)
# Note: var_map var_idx is remapped to 0..n_vars-1 for A construction.
# If you need to preserve original Flow index ordering exactly, tell me and I’ll adapt.

print(f"Mass-balance constraints: {A_bal.shape[0]} (internal nodes used: {len(mb_nodes)})")
print(f"Total replicate observations used in likelihood: {len(rep_y)}")

# -------------------------
# 3) PyMC model + sampling
# -------------------------
coords = {"flow": np.arange(n_vars), "mb": np.arange(A_bal.shape[0])}

with pm.Model(coords=coords) as model:
    typical_scale = float(np.nanmedian(np.abs(y_median[obs_mask]))) if np.any(obs_mask) else 1.0
    typical_scale = max(typical_scale, 1.0)

    x_list = []
    for j in range(n_vars):
        if obs_mask[j] and np.isfinite(sigma_obs[j]) and sigma_obs[j] > 0:
            mu0 = float(y_median[j])
            sd0 = float(max(sigma_obs[j], DEFAULT_SIGMA_FLOOR))
            lower = float(max(x_lb[j], MIN_POSITIVE)) if USE_HARD_BOUNDS else MIN_POSITIVE
            upper = float(x_ub[j]) if (USE_HARD_BOUNDS and np.isfinite(x_ub[j])) else np.inf
            xj = pm.TruncatedNormal(f"x[{j}]", mu=mu0, sigma=sd0, lower=lower, upper=upper)
        else:
            xj = pm.HalfNormal(f"x[{j}]", sigma=typical_scale)
        x_list.append(xj)

    x = pt.stack(x_list)
    pm.Deterministic("x", x, dims=("flow",))

    # Replicate likelihood: y_r ~ Normal(x[pos_r], sigma_obs[pos_r])
    pm.Normal(
        "y_like",
        mu=x[rep_pos],
        sigma=sigma_obs[rep_pos],
        observed=rep_y,
    )

    # Soft mass balance
    if A_bal.shape[0] > 0:
        Ax = pt.dot(pt.as_tensor_variable(A_bal), x)
        pm.Normal("mass_balance", mu=0.0, sigma=SIGMA_BALANCE, observed=Ax, dims=("mb",))

    trace = pm.sample(
        draws=N_DRAWS,
        tune=N_TUNE,
        chains=N_CHAINS,
        cores=min(4, N_CHAINS),
        target_accept=0.9,
        random_seed=SEED,
        progressbar=True,
    )

idata = az.from_pymc(trace, model=model)

# -------------------------
# 4) Save outputs
# -------------------------
os.makedirs(out_dir, exist_ok=True)

summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
summ.to_csv(os.path.join(out_dir, f"{out_prefix}_pymc_summary.csv"))

post = idata.posterior["x"]  # chain, draw, flow
x_mean = post.mean(dim=("chain", "draw")).to_numpy()

pm_df = var_map.copy()
pm_df["y_median"] = y_median
pm_df["sigma_obs_used"] = sigma_obs
pm_df["lower_bound"] = x_lb
pm_df["upper_bound"] = x_ub
pm_df["posterior_mean"] = x_mean
pm_df.to_csv(os.path.join(out_dir, f"{out_prefix}_posterior_mean.csv"), index=False)

# Mass-balance residual on posterior mean
if A_bal.shape[0] > 0:
    resid = A_bal @ x_mean
    print(
        "Mass-balance residuals on posterior mean:",
        "min", float(resid.min()),
        "max", float(resid.max()),
        "L2", float(np.linalg.norm(resid)),
    )

az.to_netcdf(idata, os.path.join(out_dir, f"{out_prefix}.nc"))

flat = post.stack(sample=("chain", "draw")).transpose("flow", "sample").to_numpy()
for j in range(n_vars):
    vals = flat[j, :]
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=50, density=True, alpha=0.6, color="steelblue")
    mean = float(np.mean(vals))
    low, high = np.quantile(vals, [0.025, 0.975])
    plt.axvline(mean, color="k", linestyle="--", label=f"mean = {mean:.3g}")
    plt.axvline(low, color="red", linestyle=":", label=f"2.5% = {low:.3g}")
    plt.axvline(high, color="red", linestyle=":", label=f"97.5% = {high:.3g}")
    title = f"Posterior of flow {j}"
    if "from_node_name" in var_map.columns and "to_node_name" in var_map.columns:
        title += f" ({var_map.loc[j,'from_node_name']} -> {var_map.loc[j,'to_node_name']})"
    plt.title(title)
    plt.xlabel("flow")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_{j}.png"))
    plt.close()

print(f"Saved outputs to: {out_dir}")
