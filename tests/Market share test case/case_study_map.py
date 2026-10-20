#!/usr/bin/env python3
"""
case_study_map.py
MAP estimation for the three-layer case study via constrained QP.

Minimise:
    (1/2) \Sigma_f [(x_f - μ_f)² / σ_f²]
    + (1/2) [(\Sigma x_L1→L2 - T_L2_obs)² / σ_T2²]
    + (1/2) [(\Sigma x_L2→L3 - T_L3_obs)² / σ_T3²]

Subject to:
    Mass balance at L2 nodes (with yield)
    Capacity / share constraints
    Non-negativity
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize

# ── Load data ───────────────────────────────────────────────────────
DATA_DIR = "case_study_data"
OUT_DIR = "case_study_results/map"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "flows.csv"))
with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
    truth = json.load(f)
with open(os.path.join(DATA_DIR, "constraints.json")) as f:
    cons_params = json.load(f)

N_L1, N_L2, N_L3 = 10, 3, 5
N_ARCS_12 = N_L1 * N_L2  # 30
N_ARCS_23 = N_L2 * N_L3  # 15
N_ARCS = N_ARCS_12 + N_ARCS_23  # 45

T_L2_obs = truth["T_L2_obs"]
T_L3_obs = truth["T_L3_obs"]
sigma_T2 = 50.0
sigma_T3 = 40.0

w_L2_obs = np.array(truth["w_L2_obs"])
w_L3_obs = np.array(truth["w_L3_obs"])

# ── Arc indexing ────────────────────────────────────────────────────
# Arcs 0..29: L1→L2 (source k, target m) → index = k*N_L2 + m
# Arcs 30..44: L2→L3 (source m, target d) → index = N_ARCS_12 + m*N_L3 + d

def idx_12(k, m):
    return k * N_L2 + m

def idx_23(m, d):
    return N_ARCS_12 + m * N_L3 + d

# ── Observation arrays ──────────────────────────────────────────────
obs_values = df["obs_value"].to_numpy(dtype=float)
obs_sigma = df["obs_sigma"].to_numpy(dtype=float)
is_obs = df["is_observed"].to_numpy(dtype=bool)

# For unobserved flows, use a weak prior centred at a reasonable guess
# Guess: uniform allocation from each source
typical_flow = T_L2_obs / N_ARCS_12  # ~34 kt/y per L1→L2 arc
typical_flow_23 = T_L3_obs / N_ARCS_23  # ~63 kt/y per L2→L3 arc

mu_prior = np.zeros(N_ARCS)
sigma_prior = np.full(N_ARCS, np.inf)

for f in range(N_ARCS):
    if is_obs[f]:
        mu_prior[f] = obs_values[f]
        sigma_prior[f] = obs_sigma[f]
    else:
        if f < N_ARCS_12:
            mu_prior[f] = typical_flow
            sigma_prior[f] = 2.0 * typical_flow  # very wide
        else:
            mu_prior[f] = typical_flow_23
            sigma_prior[f] = 2.0 * typical_flow_23

# ── Objective function ──────────────────────────────────────────────
def objective(x):
    """Weighted sum of squared deviations + aggregate throughput terms."""
    # Individual flow terms
    residuals = (x - mu_prior) / sigma_prior
    obj = 0.5 * np.sum(residuals**2)

    # Aggregate throughput: total into L2
    total_L2 = x[:N_ARCS_12].sum()
    obj += 0.5 * ((total_L2 - T_L2_obs) / sigma_T2)**2

    # Aggregate throughput: total into L3
    total_L3 = x[N_ARCS_12:].sum()
    obj += 0.5 * ((total_L3 - T_L3_obs) / sigma_T3)**2

    # Allocation penalty for L2 shares
    inflow_L2 = np.array([x[idx_12(k, m)]
                          for m in range(N_L2)
                          for k in range(N_L1)]).reshape(N_L2, N_L1).sum(axis=1)
    total_in_L2 = inflow_L2.sum()
    if total_in_L2 > 0:
        w_L2_model = inflow_L2 / total_in_L2
        # Dirichlet-like penalty: -\Sigma (α-1) log(w)
        # Approximate as Gaussian on shares
        kappa_L2 = 20.0
        sigma_w2 = 1.0 / (kappa_L2 * total_in_L2)
        obj += 0.5 * np.sum((w_L2_model - w_L2_obs)**2 / sigma_w2)

    # Allocation penalty for L3 shares
    inflow_L3 = np.array([x[idx_23(m, d)]
                          for d in range(N_L3)
                          for m in range(N_L2)]).reshape(N_L3, N_L2).sum(axis=1)
    total_in_L3 = inflow_L3.sum()
    if total_in_L3 > 0:
        w_L3_model = inflow_L3 / total_in_L3
        kappa_L3 = 15.0
        sigma_w3 = 1.0 / (kappa_L3 * total_in_L3)
        obj += 0.5 * np.sum((w_L3_model - w_L3_obs)**2 / sigma_w3)

    return obj


def objective_grad(x):
    """Numerical gradient (for SLSQP)."""
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = objective(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (objective(x_plus) - f0) / eps
    return grad


# ── Constraints ─────────────────────────────────────────────────────
constraints_list = []

# 1. Mass balance at each L2 node (with yield):
#    \Sigma_k x_{S_k→M_m} * yield_m - \Sigma_d x_{M_m→D_d} = 0
#    We use yield = 0.95 as a nominal value (unknown exactly)
nominal_yield = 0.95

for m in range(N_L2):
    def mb_L2(x, m=m):
        inflow = sum(x[idx_12(k, m)] for k in range(N_L1))
        outflow = sum(x[idx_23(m, d)] for d in range(N_L3))
        return inflow * nominal_yield - outflow
    constraints_list.append({"type": "eq", "fun": mb_L2})

# 2. S1→M1 ≤ 0.30 * total_L2
def cap_S1_M1(x):
    total_L2 = x[:N_ARCS_12].sum()
    return 0.30 * total_L2 - x[idx_12(0, 0)]
constraints_list.append({"type": "ineq", "fun": cap_S1_M1})

# 3. \Sigma_k x_{S_k→M2} ≥ 0.20 * total_L2
def min_M2(x):
    inflow_M2 = sum(x[idx_12(k, 1)] for k in range(N_L1))
    total_L2 = x[:N_ARCS_12].sum()
    return inflow_M2 - 0.20 * total_L2
constraints_list.append({"type": "ineq", "fun": min_M2})

# 4. \Sigma_m x_{M_m→D1} ≤ 300
def cap_D1(x):
    inflow_D1 = sum(x[idx_23(m, 0)] for m in range(N_L2))
    return 300.0 - inflow_D1
constraints_list.append({"type": "ineq", "fun": cap_D1})

# 5. Yield constraint: outflow ≤ 0.98 * inflow at each L2 node
for m in range(N_L2):
    def yield_cap(x, m=m):
        inflow = sum(x[idx_12(k, m)] for k in range(N_L1))
        outflow = sum(x[idx_23(m, d)] for d in range(N_L3))
        return 0.98 * inflow - outflow
    constraints_list.append({"type": "ineq", "fun": yield_cap})

# ── Bounds (non-negativity) ─────────────────────────────────────────
bounds = [(0, None)] * N_ARCS

# ── Initial guess ───────────────────────────────────────────────────
x0 = np.where(is_obs, obs_values, mu_prior)
x0 = np.maximum(x0, 1.0)  # ensure positive

# ── Solve ───────────────────────────────────────────────────────────
print("Solving MAP (SLSQP)...")
result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints_list,
    options={"maxiter": 2000, "ftol": 1e-12, "disp": True},
)

x_map = result.x
print(f"\nOptimisation success: {result.success}")
print(f"Objective value: {result.fun:.4f}")

# ── Evaluate solution ───────────────────────────────────────────────
x_true = df["true_value"].to_numpy(dtype=float)

# Reconstruct aggregates
total_L2_map = x_map[:N_ARCS_12].sum()
total_L3_map = x_map[N_ARCS_12:].sum()

inflow_L2_map = np.array([
    sum(x_map[idx_12(k, m)] for k in range(N_L1))
    for m in range(N_L2)
])
w_L2_map = inflow_L2_map / inflow_L2_map.sum()

inflow_L3_map = np.array([
    sum(x_map[idx_23(m, d)] for m in range(N_L2))
    for d in range(N_L3)
])
w_L3_map = inflow_L3_map / inflow_L3_map.sum()

# Mass balance residuals
mb_resid = []
for m in range(N_L2):
    inflow = sum(x_map[idx_12(k, m)] for k in range(N_L1))
    outflow = sum(x_map[idx_23(m, d)] for d in range(N_L3))
    mb_resid.append(inflow * nominal_yield - outflow)

# Error metrics
rmse = np.sqrt(np.mean((x_map - x_true)**2))
mape = np.mean(np.abs(x_map - x_true) / np.maximum(x_true, 1e-6)) * 100

print(f"\n{'='*60}")
print(f"MAP RESULTS")
print(f"{'='*60}")
print(f"Total L2: true={truth['T_L2_true']:.1f}, "
      f"obs={T_L2_obs:.1f}, MAP={total_L2_map:.1f}")
print(f"Total L3: true={truth['T_L3_true']:.1f}, "
      f"obs={T_L3_obs:.1f}, MAP={total_L3_map:.1f}")
print(f"\nL2 allocation:")
print(f"  True: {np.round(truth['w_L2_agg_true'], 3)}")
print(f"  Obs:  {np.round(w_L2_obs, 3)}")
print(f"  MAP:  {np.round(w_L2_map, 3)}")
print(f"\nL3 allocation:")
print(f"  True: {np.round(truth['w_L3_agg_true'], 3)}")
print(f"  Obs:  {np.round(w_L3_obs, 3)}")
print(f"  MAP:  {np.round(w_L3_map, 3)}")
print(f"\nMass balance residuals at L2: {np.round(mb_resid, 4)}")
print(f"RMSE (all flows): {rmse:.2f} kt/y")
print(f"MAPE (all flows): {mape:.1f}%")

# ── Check constraint satisfaction ───────────────────────────────────
print(f"\nConstraint checks:")
print(f"  S1→M1 share: {x_map[idx_12(0,0)]/total_L2_map:.3f} "
      f"(≤ 0.30: {'OK' if x_map[idx_12(0,0)] <= 0.30*total_L2_map + 1e-6 else 'VIOLATED'})")
print(f"  M2 share: {inflow_L2_map[1]/total_L2_map:.3f} "
      f"(≥ 0.20: {'OK' if inflow_L2_map[1] >= 0.20*total_L2_map - 1e-6 else 'VIOLATED'})")
print(f"  D1 inflow: {inflow_L3_map[0]:.1f} "
      f"(≤ 300: {'OK' if inflow_L3_map[0] <= 300 + 1e-6 else 'VIOLATED'})")
for m in range(N_L2):
    inflow = sum(x_map[idx_12(k, m)] for k in range(N_L1))
    outflow = sum(x_map[idx_23(m, d)] for d in range(N_L3))
    print(f"  M{m+1} yield: {outflow/inflow:.4f} "
          f"(≤ 0.98: {'OK' if outflow <= 0.98*inflow + 1e-6 else 'VIOLATED'})")

# ── Save ────────────────────────────────────────────────────────────
df_out = df.copy()
df_out["map_value"] = x_map
df_out["map_error"] = x_map - x_true
df_out["map_rel_error"] = (x_map - x_true) / np.maximum(x_true, 1e-6)
df_out.to_csv(os.path.join(OUT_DIR, "map_solution.csv"), index=False)
print(f"\nMAP solution saved to {OUT_DIR}/map_solution.csv")