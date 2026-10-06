#!/usr/bin/env python3
"""
case_study_data.py
Generate synthetic ground-truth data for the three-layer case study.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(2024)

# ── System dimensions ───────────────────────────────────────────────
N_L1 = 10   # source nodes
N_L2 = 3    # intermediate nodes
N_L3 = 5    # destination nodes
N_ARCS_12 = N_L1 * N_L2  # 30
N_ARCS_23 = N_L2 * N_L3  # 15
N_ARCS = N_ARCS_12 + N_ARCS_23  # 45

# Node numbering: L1 = 1..10, L2 = 11..13, L3 = 14..18
L1_NODES = list(range(1, N_L1 + 1))
L2_NODES = list(range(N_L1 + 1, N_L1 + N_L2 + 1))  # 11, 12, 13
L3_NODES = list(range(N_L1 + N_L2 + 1, N_L1 + N_L2 + N_L3 + 1))  # 14..18

# ── Ground truth: source totals ─────────────────────────────────────
# Each source produces a different amount
source_totals_true = np.array([
    150, 120, 100, 90, 80, 70, 110, 95, 105, 80
], dtype=float)  # sum = 1000

T_L2_true = source_totals_true.sum()  # 1000
print(f"True total into Layer 2: {T_L2_true:.1f} kt/y")

# ── Ground truth: L1 → L2 allocation per source ────────────────────
# Each source has its own allocation to the 3 intermediate nodes
# True allocation shares (10 × 3 matrix)
w_L1_to_L2_true = np.zeros((N_L1, N_L2))
alpha_base = np.array([4.5, 3.0, 2.5])  # base Dirichlet parameter
for k in range(N_L1):
    # Slightly different allocation per source
    alpha_k = alpha_base + np.random.uniform(-0.5, 0.5, size=N_L2)
    alpha_k = np.maximum(alpha_k, 0.5)
    w_L1_to_L2_true[k] = np.random.dirichlet(alpha_k)

# True L1→L2 flows (10 × 3 matrix)
x_12_true = source_totals_true[:, None] * w_L1_to_L2_true
print(f"True L1→L2 flow matrix shape: {x_12_true.shape}")
print(f"True inflow to each L2 node: {x_12_true.sum(axis=0)}")

# ── Ground truth: processing loss at Layer 2 ───────────────────────
# Each L2 node has a yield between 0.93 and 0.98
yields_L2_true = np.array([0.96, 0.94, 0.97])
inflow_L2_true = x_12_true.sum(axis=0)  # (3,)
outflow_L2_true = inflow_L2_true * yields_L2_true
T_L3_true = outflow_L2_true.sum()
print(f"True total into Layer 3: {T_L3_true:.1f} kt/y")

# ── Ground truth: L2 → L3 allocation per intermediate node ─────────
w_L2_to_L3_true = np.zeros((N_L2, N_L3))
alpha_L3_base = np.array([2.5, 2.0, 2.0, 2.0, 1.5])
for m in range(N_L2):
    alpha_m = alpha_L3_base + np.random.uniform(-0.3, 0.3, size=N_L3)
    alpha_m = np.maximum(alpha_m, 0.3)
    w_L2_to_L3_true[m] = np.random.dirichlet(alpha_m)

# True L2→L3 flows (3 × 5 matrix)
x_23_true = outflow_L2_true[:, None] * w_L2_to_L3_true
print(f"True L2→L3 flow matrix shape: {x_23_true.shape}")
print(f"True inflow to each L3 node: {x_23_true.sum(axis=0)}")

# ── Aggregate true allocations (observed as compositional data) ─────
# Observed L2 allocation = fraction of total L2 inflow going to each M_m
w_L2_agg_true = inflow_L2_true / inflow_L2_true.sum()
print(f"True L2 aggregate allocation: {w_L2_agg_true}")

# Observed L3 allocation = fraction of total L3 inflow going to each D_d
inflow_L3_true = x_23_true.sum(axis=0)
w_L3_agg_true = inflow_L3_true / inflow_L3_true.sum()
print(f"True L3 aggregate allocation: {w_L3_agg_true}")

# ── Generate noisy observations ─────────────────────────────────────
# 1. Total throughput observations
T_L2_obs = T_L2_true + np.random.normal(0, 50)  # σ = 50
T_L3_obs = T_L3_true + np.random.normal(0, 40)  # σ = 40

# 2. Allocation observations (noisy Dirichlet draws)
w_L2_obs = np.random.dirichlet(20 * w_L2_agg_true)
w_L3_obs = np.random.dirichlet(15 * w_L3_agg_true)

# 3. Individual flow observations (subset: observe ~60% of arcs)
observed_mask_12 = np.random.choice([True, False], size=(N_L1, N_L2),
                                     p=[0.6, 0.4])
observed_mask_23 = np.random.choice([True, False], size=(N_L2, N_L3),
                                     p=[0.6, 0.4])

# Noisy observations with 15% relative uncertainty
x_12_obs = np.where(observed_mask_12,
                     x_12_true * (1 + np.random.normal(0, 0.15,
                                                        size=x_12_true.shape)),
                     np.nan)
x_23_obs = np.where(observed_mask_23,
                     x_23_true * (1 + np.random.normal(0, 0.15,
                                                        size=x_23_true.shape)),
                     np.nan)

# ── Build flow table ────────────────────────────────────────────────
records = []
arc_idx = 0

# L1 → L2 arcs
for k in range(N_L1):
    for m in range(N_L2):
        src = L1_NODES[k]
        tgt = L2_NODES[m]
        records.append({
            "flow_idx": arc_idx,
            "from_node": src,
            "to_node": tgt,
            "layer": "L1_to_L2",
            "true_value": x_12_true[k, m],
            "obs_value": x_12_obs[k, m] if observed_mask_12[k, m] else np.nan,
            "obs_sigma": 0.15 * x_12_true[k, m] if observed_mask_12[k, m] else np.nan,
            "is_observed": observed_mask_12[k, m],
        })
        arc_idx += 1

# L2 → L3 arcs
for m in range(N_L2):
    for d in range(N_L3):
        src = L2_NODES[m]
        tgt = L3_NODES[d]
        records.append({
            "flow_idx": arc_idx,
            "from_node": src,
            "to_node": tgt,
            "layer": "L2_to_L3",
            "true_value": x_23_true[m, d],
            "obs_value": x_23_obs[m, d] if observed_mask_23[m, d] else np.nan,
            "obs_sigma": 0.15 * x_23_true[m, d] if observed_mask_23[m, d] else np.nan,
            "is_observed": observed_mask_23[m, d],
        })
        arc_idx += 1

df_flows = pd.DataFrame(records)

# ── Save everything ─────────────────────────────────────────────────
OUT_DIR = "case_study_data"
os.makedirs(OUT_DIR, exist_ok=True)

df_flows.to_csv(os.path.join(OUT_DIR, "flows.csv"), index=False)

# Save ground truth
truth = {
    "T_L2_true": T_L2_true,
    "T_L3_true": T_L3_true,
    "T_L2_obs": T_L2_obs,
    "T_L3_obs": T_L3_obs,
    "w_L2_agg_true": w_L2_agg_true.tolist(),
    "w_L3_agg_true": w_L3_agg_true.tolist(),
    "w_L2_obs": w_L2_obs.tolist(),
    "w_L3_obs": w_L3_obs.tolist(),
    "yields_L2_true": yields_L2_true.tolist(),
    "inflow_L2_true": inflow_L2_true.tolist(),
    "outflow_L2_true": outflow_L2_true.tolist(),
}

import json
with open(os.path.join(OUT_DIR, "ground_truth.json"), "w") as f:
    json.dump(truth, f, indent=2)

# Save constraint parameters
constraints = {
    "max_S1_to_M1_share": 0.30,
    "min_M2_share": 0.20,
    "max_D1_inflow": 300.0,
    "max_yield_L2": 0.98,
}
with open(os.path.join(OUT_DIR, "constraints.json"), "w") as f:
    json.dump(constraints, f, indent=2)

print(f"\n{'='*60}")
print(f"Synthetic data saved to {OUT_DIR}/")
print(f"  Arcs: {N_ARCS} ({N_ARCS_12} L1→L2 + {N_ARCS_23} L2→L3)")
print(f"  Observed arcs: {df_flows['is_observed'].sum()} / {N_ARCS}")
print(f"  T_L2 true={T_L2_true:.1f}, obs={T_L2_obs:.1f}")
print(f"  T_L3 true={T_L3_true:.1f}, obs={T_L3_obs:.1f}")
print(f"  w_L2 true={np.round(w_L2_agg_true, 3)}, obs={np.round(w_L2_obs, 3)}")
print(f"  w_L3 true={np.round(w_L3_agg_true, 3)}, obs={np.round(w_L3_obs, 3)}")