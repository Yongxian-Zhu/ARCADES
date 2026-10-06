#!/usr/bin/env python3
"""
case_study_compare.py
Compare the three inference approaches on the synthetic case study.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_DIR = "case_study_data"
MAP_DIR = "case_study_results/map"
FLOW_DIR = "case_study_results/flow_mcmc"
ALLOC_DIR = "case_study_results/alloc_mcmc"
OUT_DIR = "case_study_results/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
    truth = json.load(f)

# ── Load results ────────────────────────────────────────────────────
df_map = pd.read_csv(os.path.join(MAP_DIR, "map_solution.csv"))
df_flow = pd.read_csv(os.path.join(FLOW_DIR, "flow_mcmc_results.csv"))
df_alloc = pd.read_csv(os.path.join(ALLOC_DIR, "alloc_mcmc_results.csv"))

x_true = df_map["true_value"].to_numpy()
x_map = df_map["map_value"].to_numpy()
x_flow_mean = df_flow["posterior_mean"].to_numpy()
x_flow_std = df_flow["posterior_std"].to_numpy()
x_alloc_mean = df_alloc["posterior_mean"].to_numpy()
x_alloc_std = df_alloc["posterior_std"].to_numpy()

N_ARCS = len(x_true)

# ── Error metrics ───────────────────────────────────────────────────
def compute_metrics(x_est, x_true, x_std=None, x_lo=None, x_hi=None):
    rmse = np.sqrt(np.mean((x_est - x_true)**2))
    mape = np.mean(np.abs(x_est - x_true) / np.maximum(x_true, 1e-6)) * 100
    max_err = np.max(np.abs(x_est - x_true))
    bias = np.mean(x_est - x_true)
    metrics = {"RMSE": rmse, "MAPE": mape, "MaxErr": max_err, "Bias": bias}
    if x_std is not None:
        metrics["MeanStd"] = np.mean(x_std)
    if x_lo is not None and x_hi is not None:
        coverage = np.mean((x_true >= x_lo) & (x_true <= x_hi))
        ci_width = np.mean(x_hi - x_lo)
        metrics["Coverage95"] = coverage
        metrics["MeanCIWidth"] = ci_width
    return metrics

metrics_map = compute_metrics(x_map, x_true)
metrics_flow = compute_metrics(
    x_flow_mean, x_true, x_flow_std,
    df_flow["ci_lower"].to_numpy(), df_flow["ci_upper"].to_numpy()
)
metrics_alloc = compute_metrics(
    x_alloc_mean, x_true, x_alloc_std,
    df_alloc["ci_lower"].to_numpy(), df_alloc["ci_upper"].to_numpy()
)

# ── Print comparison table ──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"COMPARISON OF THREE INFERENCE APPROACHES")
print(f"{'='*70}")
print(f"\n{'Metric':<20} {'MAP':>12} {'Flow MCMC':>12} {'Alloc MCMC':>12}")
print(f"{'-'*56}")
for key in ["RMSE", "MAPE", "MaxErr", "Bias", "MeanStd",
            "Coverage95", "MeanCIWidth"]:
    v_map = metrics_map.get(key, "—")
    v_flow = metrics_flow.get(key, "—")
    v_alloc = metrics_alloc.get(key, "—")
    fmt = lambda v: f"{v:>12.2f}" if isinstance(v, float) else f"{v:>12}"
    if key == "Coverage95":
        fmt = lambda v: f"{v:>11.1%}" if isinstance(v, float) else f"{v:>12}"
    print(f"{key:<20} {fmt(v_map)} {fmt(v_flow)} {fmt(v_alloc)}")

# ── Observed vs unobserved breakdown ────────────────────────────────
is_obs = df_map["is_observed"].to_numpy(dtype=bool)
print(f"\n{'='*70}")
print(f"BREAKDOWN: OBSERVED vs UNOBSERVED FLOWS")
print(f"{'='*70}")
for label, mask in [("Observed", is_obs), ("Unobserved", ~is_obs)]:
    n = mask.sum()
    rmse_map = np.sqrt(np.mean((x_map[mask] - x_true[mask])**2))
    rmse_flow = np.sqrt(np.mean((x_flow_mean[mask] - x_true[mask])**2))
    rmse_alloc = np.sqrt(np.mean((x_alloc_mean[mask] - x_true[mask])**2))
    print(f"\n{label} flows (n={n}):")
    print(f"  RMSE:  MAP={rmse_map:.2f}, Flow={rmse_flow:.2f}, "
          f"Alloc={rmse_alloc:.2f}")
    if "ci_lower" in df_flow.columns:
        cov_flow = np.mean(
            (x_true[mask] >= df_flow["ci_lower"].to_numpy()[mask]) &
            (x_true[mask] <= df_flow["ci_upper"].to_numpy()[mask])
        )
        cov_alloc = np.mean(
            (x_true[mask] >= df_alloc["ci_lower"].to_numpy()[mask]) &
            (x_true[mask] <= df_alloc["ci_upper"].to_numpy()[mask])
        )
        print(f"  95% CI coverage: Flow={cov_flow:.1%}, Alloc={cov_alloc:.1%}")

# ── Figure 1: True vs Estimated scatter ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, x_est, x_sd, title, color in [
    (axes[0], x_map, None, "MAP (QP)", "tab:blue"),
    (axes[1], x_flow_mean, x_flow_std, "Flow-Based MCMC", "tab:orange"),
    (axes[2], x_alloc_mean, x_alloc_std, "Node–Allocation MCMC", "tab:green"),
]:
    # Observed vs unobserved
    ax.scatter(x_true[is_obs], x_est[is_obs],
               alpha=0.6, s=30, color=color, label="Observed", zorder=3)
    ax.scatter(x_true[~is_obs], x_est[~is_obs],
               alpha=0.4, s=30, color=color, marker="^",
               edgecolors="grey", label="Unobserved", zorder=3)

    if x_sd is not None:
        # Error bars for MCMC
        ax.errorbar(x_true, x_est, yerr=1.96 * x_sd,
                     fmt="none", ecolor="lightgrey", alpha=0.3, zorder=1)

    # 1:1 line
    lim = max(x_true.max(), x_est.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True flow (kt/y)", fontsize=11)
    ax.set_ylabel("Estimated flow (kt/y)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    # Add RMSE annotation
    rmse = np.sqrt(np.mean((x_est - x_true)**2))
    ax.text(0.05, 0.92, f"RMSE = {rmse:.1f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.suptitle("Case Study: True vs Estimated Flows", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_comparison.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ── Figure 2: Uncertainty comparison ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Posterior std comparison
ax = axes[0]
arc_ids = np.arange(N_ARCS)
ax.bar(arc_ids - 0.2, x_flow_std, width=0.4, alpha=0.7,
       color="tab:orange", label="Flow MCMC")
ax.bar(arc_ids + 0.2, x_alloc_std, width=0.4, alpha=0.7,
       color="tab:green", label="Alloc MCMC")
ax.set_xlabel("Arc index")
ax.set_ylabel("Posterior std (kt/y)")
ax.set_title("Posterior Uncertainty by Arc")
ax.legend()

# Panel B: CI coverage by approach
ax = axes[1]
# Compute rolling coverage (sorted by true value)
sort_idx = np.argsort(x_true)
window = 10
cov_flow_rolling = []
cov_alloc_rolling = []
x_true_sorted = x_true[sort_idx]
for i in range(len(sort_idx) - window + 1):
    idx_win = sort_idx[i:i+window]
    cov_f = np.mean(
        (x_true[idx_win] >= df_flow["ci_lower"].to_numpy()[idx_win]) &
        (x_true[idx_win] <= df_flow["ci_upper"].to_numpy()[idx_win])
    )
    cov_a = np.mean(
        (x_true[idx_win] >= df_alloc["ci_lower"].to_numpy()[idx_win]) &
        (x_true[idx_win] <= df_alloc["ci_upper"].to_numpy()[idx_win])
    )
    cov_flow_rolling.append(cov_f)
    cov_alloc_rolling.append(cov_a)

x_mid = x_true_sorted[window//2:-(window//2)]
if len(x_mid) == len(cov_flow_rolling):
    ax.plot(x_mid, cov_flow_rolling, color="tab:orange",
            label="Flow MCMC", lw=2)
    ax.plot(x_mid, cov_alloc_rolling, color="tab:green",
            label="Alloc MCMC", lw=2)
ax.axhline(0.95, color="red", ls="--", lw=1, label="Nominal 95%")
ax.set_xlabel("True flow magnitude (kt/y)")
ax.set_ylabel("Rolling 95% CI coverage")
ax.set_title("Calibration: CI Coverage vs Flow Magnitude")
ax.legend()
ax.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "uncertainty_comparison.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ── Figure 3: Allocation recovery ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# L2 allocation
ax = axes[0]
w_L2_true = np.array(truth["w_L2_agg_true"])
w_L2_obs = np.array(truth["w_L2_obs"])

# Compute MAP allocation
N_L1, N_L2_n, N_L3_n = 10, 3, 5
N_ARCS_12 = N_L1 * N_L2_n
inflow_L2_map = np.zeros(N_L2_n)
inflow_L2_flow = np.zeros(N_L2_n)
inflow_L2_alloc = np.zeros(N_L2_n)
for m in range(N_L2_n):
    for k in range(N_L1):
        fidx = k * N_L2_n + m
        inflow_L2_map[m] += x_map[fidx]
        inflow_L2_flow[m] += x_flow_mean[fidx]
        inflow_L2_alloc[m] += x_alloc_mean[fidx]
w_L2_map = inflow_L2_map / inflow_L2_map.sum()
w_L2_flow = inflow_L2_flow / inflow_L2_flow.sum()
w_L2_alloc = inflow_L2_alloc / inflow_L2_alloc.sum()

x_pos = np.arange(N_L2_n)
width = 0.15
ax.bar(x_pos - 2*width, w_L2_true, width, label="True", color="black", alpha=0.8)
ax.bar(x_pos - width, w_L2_obs, width, label="Observed", color="grey", alpha=0.6)
ax.bar(x_pos, w_L2_map, width, label="MAP", color="tab:blue", alpha=0.7)
ax.bar(x_pos + width, w_L2_flow, width, label="Flow MCMC", color="tab:orange", alpha=0.7)
ax.bar(x_pos + 2*width, w_L2_alloc, width, label="Alloc MCMC", color="tab:green", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"M{m+1}" for m in range(N_L2_n)])
ax.set_ylabel("Allocation share")
ax.set_title("Layer 2 Allocation Recovery")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.6)

# L3 allocation
ax = axes[1]
w_L3_true = np.array(truth["w_L3_agg_true"])
w_L3_obs = np.array(truth["w_L3_obs"])

inflow_L3_map = np.zeros(N_L3_n)
inflow_L3_flow = np.zeros(N_L3_n)
inflow_L3_alloc = np.zeros(N_L3_n)
for d in range(N_L3_n):
    for m in range(N_L2_n):
        fidx = N_ARCS_12 + m * N_L3_n + d
        inflow_L3_map[d] += x_map[fidx]
        inflow_L3_flow[d] += x_flow_mean[fidx]
        inflow_L3_alloc[d] += x_alloc_mean[fidx]
w_L3_map = inflow_L3_map / inflow_L3_map.sum()
w_L3_flow = inflow_L3_flow / inflow_L3_flow.sum()
w_L3_alloc = inflow_L3_alloc / inflow_L3_alloc.sum()

x_pos = np.arange(N_L3_n)
ax.bar(x_pos - 2*width, w_L3_true, width, label="True", color="black", alpha=0.8)
ax.bar(x_pos - width, w_L3_obs, width, label="Observed", color="grey", alpha=0.6)
ax.bar(x_pos, w_L3_map, width, label="MAP", color="tab:blue", alpha=0.7)
ax.bar(x_pos + width, w_L3_flow, width, label="Flow MCMC", color="tab:orange", alpha=0.7)
ax.bar(x_pos + 2*width, w_L3_alloc, width, label="Alloc MCMC", color="tab:green", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"D{d+1}" for d in range(N_L3_n)])
ax.set_ylabel("Allocation share")
ax.set_title("Layer 3 Allocation Recovery")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "allocation_comparison.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ── Summary table ───────────────────────────────────────────────────
summary_rows = []
for name, metrics in [("MAP", metrics_map),
                       ("Flow MCMC", metrics_flow),
                       ("Alloc MCMC", metrics_alloc)]:
    row = {"Method": name}
    row.update(metrics)
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "comparison_summary.csv"), index=False)

print(f"\n{'='*70}")
print(f"SUMMARY TABLE")
print(f"{'='*70}")
print(df_summary.to_string(index=False))

print(f"\nAll comparison outputs saved to {OUT_DIR}/")