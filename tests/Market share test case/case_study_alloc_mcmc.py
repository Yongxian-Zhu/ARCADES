#!/usr/bin/env python3
"""
case_study_alloc_mcmc.py
Node–allocation Bayesian inference for the three-layer case study.

Latent variables:
  - T_k (k=1..10): total output from each source node
  - w_k (k=1..10): Dirichlet allocation from source k to L2 nodes (3-simplex)
  - T_m^out (m=1..3): total output from each L2 node
  - w_m (m=1..3): Dirichlet allocation from L2 node m to L3 nodes (5-simplex)

Reconstructed flows:
  x_{S_k→M_m} = T_k · w_k[m]
  x_{M_m→D_d} = T_m^out · w_m[d]

Key advantage: simplex constraints \Sigma_j w_{ij} = 1 are enforced BY CONSTRUCTION.
"""

import numpy as np
import pandas as pd
import json
import os
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── Load data ───────────────────────────────────────────────────────
DATA_DIR = "case_study_data"
OUT_DIR = "case_study_results/alloc_mcmc"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "flows.csv"))
with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
    truth = json.load(f)
with open(os.path.join(DATA_DIR, "constraints.json")) as f:
    cons_params = json.load(f)

N_L1, N_L2, N_L3 = 10, 3, 5
N_ARCS_12 = N_L1 * N_L2
N_ARCS_23 = N_L2 * N_L3
N_ARCS = N_ARCS_12 + N_ARCS_23

T_L2_obs = truth["T_L2_obs"]
T_L3_obs = truth["T_L3_obs"]
sigma_T2 = 50.0
sigma_T3 = 40.0
w_L2_obs = np.array(truth["w_L2_obs"])
w_L3_obs = np.array(truth["w_L3_obs"])

# ── Arc indexing ────────────────────────────────────────────────────
def idx_12(k, m):
    return k * N_L2 + m

def idx_23(m, d):
    return N_ARCS_12 + m * N_L3 + d

# ── Observation arrays ──────────────────────────────────────────────
obs_values = df["obs_value"].to_numpy(dtype=float)
obs_sigma = df["obs_sigma"].to_numpy(dtype=float)
is_obs = df["is_observed"].to_numpy(dtype=bool)
x_true = df["true_value"].to_numpy(dtype=float)

# Replicate table
rep_idx = np.where(is_obs)[0]
rep_y = obs_values[rep_idx]
rep_sigma = obs_sigma[rep_idx]

# ── Compute observed source totals and allocations ──────────────────
# For each source k, compute observed total and allocation from available data
source_obs_total = np.zeros(N_L1)
source_obs_alloc = np.zeros((N_L1, N_L2))
source_has_obs = np.zeros(N_L1, dtype=bool)

for k in range(N_L1):
    obs_flows_k = []
    for m in range(N_L2):
        fidx = idx_12(k, m)
        if is_obs[fidx]:
            obs_flows_k.append((m, obs_values[fidx]))
    if len(obs_flows_k) > 0:
        source_has_obs[k] = True
        total_k = sum(v for _, v in obs_flows_k)
        source_obs_total[k] = total_k
        for m, v in obs_flows_k:
            source_obs_alloc[k, m] = v / total_k if total_k > 0 else 1.0 / N_L2
    else:
        source_obs_total[k] = T_L2_obs / N_L1  # uniform guess
        source_obs_alloc[k] = np.ones(N_L2) / N_L2

# For each L2 node m, compute observed total output and allocation
l2_obs_total = np.zeros(N_L2)
l2_obs_alloc = np.zeros((N_L2, N_L3))
l2_has_obs = np.zeros(N_L2, dtype=bool)

for m in range(N_L2):
    obs_flows_m = []
    for d in range(N_L3):
        fidx = idx_23(m, d)
        if is_obs[fidx]:
            obs_flows_m.append((d, obs_values[fidx]))
    if len(obs_flows_m) > 0:
        l2_has_obs[m] = True
        total_m = sum(v for _, v in obs_flows_m)
        l2_obs_total[m] = total_m
        for d, v in obs_flows_m:
            l2_obs_alloc[m, d] = v / total_m if total_m > 0 else 1.0 / N_L3
    else:
        l2_obs_total[m] = T_L3_obs / N_L2
        l2_obs_alloc[m] = np.ones(N_L3) / N_L3

# ── Mass balance matrix (on reconstructed x) ───────────────────────
nominal_yield = 0.95
n_mb = N_L2
A_mb = np.zeros((n_mb, N_ARCS))
for m in range(N_L2):
    for k in range(N_L1):
        A_mb[m, idx_12(k, m)] = nominal_yield
    for d in range(N_L3):
        A_mb[m, idx_23(m, d)] = -1.0

SIGMA_BAL = 5.0

# ── Dirichlet prior parameters ──────────────────────────────────────
# For source allocations: use observed shares with moderate concentration
KAPPA_SRC = 8.0   # moderate: allows flexibility
KAPPA_L2 = 10.0   # slightly stronger for L2→L3

# For aggregate allocation observations
KAPPA_AGG_L2 = 20.0
KAPPA_AGG_L3 = 15.0

def safe_alpha(shares, kappa, floor=0.3):
    """Compute Dirichlet α = \kappa·w with a floor."""
    alpha = kappa * np.array(shares)
    return np.maximum(alpha, floor)


# ── PyMC model ──────────────────────────────────────────────────────
print("Building node–allocation PyMC model...")

with pm.Model() as alloc_model:

    # ════════════════════════════════════════════════════════════════
    # Layer 1 → Layer 2: T_k and w_k for each source
    # ════════════════════════════════════════════════════════════════
    T_src = []
    w_src = []

    for k in range(N_L1):
        # Total output from source k
        if source_has_obs[k]:
            mu_T = float(source_obs_total[k])
            sigma_T = float(max(0.3 * mu_T, 5.0))
        else:
            mu_T = float(T_L2_obs / N_L1)
            sigma_T = float(0.5 * mu_T)

        T_k = pm.TruncatedNormal(
            f"T_src[{k}]",
            mu=mu_T,
            sigma=sigma_T,
            lower=1.0,
        )
        T_src.append(T_k)

        # Allocation from source k to L2 nodes
        alpha_k = safe_alpha(source_obs_alloc[k], KAPPA_SRC)
        w_k = pm.Dirichlet(f"w_src[{k}]", a=alpha_k)
        w_src.append(w_k)

    # ════════════════════════════════════════════════════════════════
    # Layer 2 → Layer 3: T_m^out and w_m for each L2 node
    # ════════════════════════════════════════════════════════════════
    T_l2_out = []
    w_l2 = []

    for m in range(N_L2):
        if l2_has_obs[m]:
            mu_T = float(l2_obs_total[m])
            sigma_T = float(max(0.3 * mu_T, 5.0))
        else:
            mu_T = float(T_L3_obs / N_L2)
            sigma_T = float(0.5 * mu_T)

        T_m = pm.TruncatedNormal(
            f"T_l2_out[{m}]",
            mu=mu_T,
            sigma=sigma_T,
            lower=1.0,
        )
        T_l2_out.append(T_m)

        # Allocation from L2 node m to L3 nodes
        alpha_m = safe_alpha(l2_obs_alloc[m], KAPPA_L2)
        w_m = pm.Dirichlet(f"w_l2[{m}]", a=alpha_m)
        w_l2.append(w_m)

    # ════════════════════════════════════════════════════════════════
    # Reconstruct all arc flows: x = T · w
    # ════════════════════════════════════════════════════════════════
    x_parts = {}

    # L1 → L2 flows
    for k in range(N_L1):
        for m in range(N_L2):
            x_parts[idx_12(k, m)] = T_src[k] * w_src[k][m]

    # L2 → L3 flows
    for m in range(N_L2):
        for d in range(N_L3):
            x_parts[idx_23(m, d)] = T_l2_out[m] * w_l2[m][d]

    # Stack into full x vector
    x = pt.stack([x_parts[f] for f in range(N_ARCS)])
    pm.Deterministic("x", x)

    # Also register T and w as deterministics for output
    T_src_vec = pt.stack(T_src)
    pm.Deterministic("T_src", T_src_vec)

    T_l2_out_vec = pt.stack(T_l2_out)
    pm.Deterministic("T_l2_out", T_l2_out_vec)

    for k in range(N_L1):
        pm.Deterministic(f"w_src_det[{k}]", w_src[k])
    for m in range(N_L2):
        pm.Deterministic(f"w_l2_det[{m}]", w_l2[m])

    # ════════════════════════════════════════════════════════════════
    # Likelihoods
    # ════════════════════════════════════════════════════════════════

    # 1. Individual flow observations
    pm.Normal(
        "y_obs",
        mu=x[rep_idx],
        sigma=pt.as_tensor_variable(rep_sigma),
        observed=rep_y,
    )

    # 2. Aggregate throughput into L2
    total_L2 = pt.sum(x[:N_ARCS_12])
    pm.Normal("T_L2_like", mu=total_L2, sigma=sigma_T2,
              observed=np.array(T_L2_obs))

    # 3. Aggregate throughput into L3
    total_L3 = pt.sum(x[N_ARCS_12:])
    pm.Normal("T_L3_like", mu=total_L3, sigma=sigma_T3,
              observed=np.array(T_L3_obs))

    # 4. Aggregate L2 allocation observation (Dirichlet likelihood)
    inflow_L2 = pt.stack([
        pt.sum(pt.stack([x[idx_12(k, m)] for k in range(N_L1)]))
        for m in range(N_L2)
    ])
    w_L2_model = inflow_L2 / pt.sum(inflow_L2)
    # Dirichlet likelihood on observed aggregate shares
    alpha_L2_obs = safe_alpha(w_L2_obs, KAPPA_AGG_L2)
    pm.Dirichlet("w_L2_agg_like", a=alpha_L2_obs, observed=w_L2_model)

    # 5. Aggregate L3 allocation observation (Dirichlet likelihood)
    inflow_L3 = pt.stack([
        pt.sum(pt.stack([x[idx_23(m, d)] for m in range(N_L2)]))
        for d in range(N_L3)
    ])
    w_L3_model = inflow_L3 / pt.sum(inflow_L3)
    alpha_L3_obs = safe_alpha(w_L3_obs, KAPPA_AGG_L3)
    pm.Dirichlet("w_L3_agg_like", a=alpha_L3_obs, observed=w_L3_model)

    # 6. Soft mass balance at L2 nodes
    Ax = pt.dot(pt.as_tensor_variable(A_mb), x)
    pm.Normal("mass_balance", mu=0.0, sigma=SIGMA_BAL, observed=Ax)

    # 7. Soft inequality constraints
    # S1→M1 ≤ 0.30 * total_L2
    slack_S1M1 = 0.30 * total_L2 - x[idx_12(0, 0)]
    pm.HalfNormal("ineq_S1M1", sigma=5.0,
                  observed=pt.maximum(slack_S1M1, 0.0))

    # M2 share ≥ 0.20
    slack_M2 = inflow_L2[1] - 0.20 * total_L2
    pm.HalfNormal("ineq_M2", sigma=5.0,
                  observed=pt.maximum(slack_M2, 0.0))

    # D1 inflow ≤ 300
    slack_D1 = 300.0 - inflow_L3[0]
    pm.HalfNormal("ineq_D1", sigma=5.0,
                  observed=pt.maximum(slack_D1, 0.0))

    # Yield constraints
    for m in range(N_L2):
        outflow_m = pt.sum(pt.stack([x[idx_23(m, d)] for d in range(N_L3)]))
        inflow_m = pt.sum(pt.stack([x[idx_12(k, m)] for k in range(N_L1)]))
        slack_yield = 0.98 * inflow_m - outflow_m
        pm.HalfNormal(f"ineq_yield_M{m+1}", sigma=2.0,
                      observed=pt.maximum(slack_yield, 0.0))

    # ════════════════════════════════════════════════════════════════
    # Sample
    # ════════════════════════════════════════════════════════════════
    print("Sampling (node–allocation MCMC)...")
    trace = pm.sample(
        draws=1000,
        tune=1500,  # extra tuning for bilinear geometry
        chains=4,
        cores=4,
        target_accept=0.95,  # higher for Dirichlet + bilinear
        random_seed=42,
        progressbar=True,
    )

idata = az.from_pymc(trace, model=alloc_model)

# ── Extract results ─────────────────────────────────────────────────
post_x = idata.posterior["x"]
x_mean = post_x.mean(dim=("chain", "draw")).to_numpy()
x_std = post_x.std(dim=("chain", "draw")).to_numpy()
x_q025 = post_x.quantile(0.025, dim=("chain", "draw")).to_numpy()
x_q975 = post_x.quantile(0.975, dim=("chain", "draw")).to_numpy()

x_samples = post_x.stack(sample=("chain", "draw")).to_numpy()

# Source throughputs
post_T_src = idata.posterior["T_src"]
T_src_mean = post_T_src.mean(dim=("chain", "draw")).to_numpy()
T_src_std = post_T_src.std(dim=("chain", "draw")).to_numpy()

# L2 output throughputs
post_T_l2 = idata.posterior["T_l2_out"]
T_l2_mean = post_T_l2.mean(dim=("chain", "draw")).to_numpy()
T_l2_std = post_T_l2.std(dim=("chain", "draw")).to_numpy()

# Source allocations
w_src_means = []
for k in range(N_L1):
    vname = f"w_src_det[{k}]"
    if vname in idata.posterior:
        w_k = idata.posterior[vname].mean(dim=("chain", "draw")).to_numpy()
    else:
        w_k = np.ones(N_L2) / N_L2
    w_src_means.append(w_k)

# L2 allocations
w_l2_means = []
for m in range(N_L2):
    vname = f"w_l2_det[{m}]"
    if vname in idata.posterior:
        w_m = idata.posterior[vname].mean(dim=("chain", "draw")).to_numpy()
    else:
        w_m = np.ones(N_L3) / N_L3
    w_l2_means.append(w_m)

# ── Aggregates ──────────────────────────────────────────────────────
total_L2_samples = x_samples[:N_ARCS_12].sum(axis=0)
total_L3_samples = x_samples[N_ARCS_12:].sum(axis=0)

inflow_L2_samples = np.zeros((N_L2, x_samples.shape[1]))
for m in range(N_L2):
    for k in range(N_L1):
        inflow_L2_samples[m] += x_samples[idx_12(k, m)]
w_L2_samples = inflow_L2_samples / inflow_L2_samples.sum(axis=0, keepdims=True)

inflow_L3_samples = np.zeros((N_L3, x_samples.shape[1]))
for d in range(N_L3):
    for m in range(N_L2):
        inflow_L3_samples[d] += x_samples[idx_23(m, d)]
w_L3_samples = inflow_L3_samples / inflow_L3_samples.sum(axis=0, keepdims=True)

# ── Print results ───────────────────────────────────────────────────
rmse = np.sqrt(np.mean((x_mean - x_true)**2))
mape = np.mean(np.abs(x_mean - x_true) / np.maximum(x_true, 1e-6)) * 100
coverage = np.mean((x_true >= x_q025) & (x_true <= x_q975))

print(f"\n{'='*60}")
print(f"NODE–ALLOCATION MCMC RESULTS")
print(f"{'='*60}")

print(f"\nSource throughputs (T_k):")
# Compute true source totals
source_totals_true = np.array([
    sum(x_true[idx_12(k, m)] for m in range(N_L2))
    for k in range(N_L1)
])
for k in range(N_L1):
    print(f"  S{k+1}: true={source_totals_true[k]:.1f}, "
          f"mean={T_src_mean[k]:.1f} ± {T_src_std[k]:.1f}")

print(f"\nL2 output throughputs (T_m^out):")
l2_out_true = np.array(truth["outflow_L2_true"])
for m in range(N_L2):
    print(f"  M{m+1}: true={l2_out_true[m]:.1f}, "
          f"mean={T_l2_mean[m]:.1f} ± {T_l2_std[m]:.1f}")

print(f"\nSource allocation vectors (w_k):")
for k in range(min(3, N_L1)):  # show first 3
    w_true_k = np.array([x_true[idx_12(k, m)] for m in range(N_L2)])
    w_true_k = w_true_k / w_true_k.sum()
    print(f"  S{k+1}: true={np.round(w_true_k, 3)}, "
          f"mean={np.round(w_src_means[k], 3)}")

print(f"\nL2 allocation vectors (w_m):")
for m in range(N_L2):
    w_true_m = np.array([x_true[idx_23(m, d)] for d in range(N_L3)])
    w_true_m = w_true_m / w_true_m.sum()
    print(f"  M{m+1}: true={np.round(w_true_m, 3)}, "
          f"mean={np.round(w_l2_means[m], 3)}")

print(f"\nAggregate L2 allocation (mean [95% CI]):")
w_L2_true = np.array(truth["w_L2_agg_true"])
for m in range(N_L2):
    print(f"  M{m+1}: true={w_L2_true[m]:.3f}, "
          f"mean={np.mean(w_L2_samples[m]):.3f} "
          f"[{np.quantile(w_L2_samples[m], 0.025):.3f}, "
          f"{np.quantile(w_L2_samples[m], 0.975):.3f}]")

print(f"\nAggregate L3 allocation (mean [95% CI]):")
w_L3_true = np.array(truth["w_L3_agg_true"])
for d in range(N_L3):
    print(f"  D{d+1}: true={w_L3_true[d]:.3f}, "
          f"mean={np.mean(w_L3_samples[d]):.3f} "
          f"[{np.quantile(w_L3_samples[d], 0.025):.3f}, "
          f"{np.quantile(w_L3_samples[d], 0.975):.3f}]")

print(f"\nTotal L2: true={truth['T_L2_true']:.1f}, "
      f"mean={np.mean(total_L2_samples):.1f} "
      f"[{np.quantile(total_L2_samples, 0.025):.1f}, "
      f"{np.quantile(total_L2_samples, 0.975):.1f}]")
print(f"Total L3: true={truth['T_L3_true']:.1f}, "
      f"mean={np.mean(total_L3_samples):.1f} "
      f"[{np.quantile(total_L3_samples, 0.025):.1f}, "
      f"{np.quantile(total_L3_samples, 0.975):.1f}]")

print(f"\nRMSE (all flows): {rmse:.2f} kt/y")
print(f"MAPE (all flows): {mape:.1f}%")
print(f"95% CI coverage: {coverage:.1%}")

# Simplex check: do allocations sum to 1?
for k in range(N_L1):
    w_sum = w_src_means[k].sum()
    if abs(w_sum - 1.0) > 1e-6:
        print(f"  WARNING: w_src[{k}] sums to {w_sum:.6f}")
for m in range(N_L2):
    w_sum = w_l2_means[m].sum()
    if abs(w_sum - 1.0) > 1e-6:
        print(f"  WARNING: w_l2[{m}] sums to {w_sum:.6f}")
print("Simplex constraints satisfied by construction ✓")

# Mass balance residuals
mb_resid = A_mb @ x_mean
print(f"Mass balance residuals: max|r|={np.max(np.abs(mb_resid)):.4f}")

# Convergence
summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
rhat_max = summ["r_hat"].max()
ess_min = summ["ess_bulk"].min()
print(f"Convergence: max R-hat={rhat_max:.4f}, min ESS={ess_min:.0f}")

# ── Save ────────────────────────────────────────────────────────────
df_out = df.copy()
df_out["posterior_mean"] = x_mean
df_out["posterior_std"] = x_std
df_out["ci_lower"] = x_q025
df_out["ci_upper"] = x_q975
df_out["error"] = x_mean - x_true
df_out["in_ci"] = (x_true >= x_q025) & (x_true <= x_q975)
df_out.to_csv(os.path.join(OUT_DIR, "alloc_mcmc_results.csv"), index=False)

az.to_netcdf(idata, os.path.join(OUT_DIR, "alloc_mcmc_trace.nc"))
summ.to_csv(os.path.join(OUT_DIR, "alloc_mcmc_summary.csv"))

# ── Allocation posterior plots ──────────────────────────────────────
# L2 allocation simplex (ternary-like: bar charts)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for k in range(N_L1):
    ax = axes[k // 5, k % 5]
    vname = f"w_src_det[{k}]"
    if vname in idata.posterior:
        w_post = idata.posterior[vname].stack(
            sample=("chain", "draw")).to_numpy()  # (3, n_samples)
        positions = np.arange(N_L2)
        w_true_k = np.array([x_true[idx_12(k, m)] for m in range(N_L2)])
        w_true_k = w_true_k / w_true_k.sum()
        bp = ax.boxplot([w_post[m] for m in range(N_L2)],
                        positions=positions, widths=0.6,
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.scatter(positions, w_true_k, color="red", zorder=5,
                   s=50, marker="D", label="True")
    ax.set_title(f"S{k+1} → L2", fontsize=9)
    ax.set_xticks(range(N_L2))
    ax.set_xticklabels([f"M{m+1}" for m in range(N_L2)], fontsize=7)
    ax.set_ylim(0, 1)
    if k == 0:
        ax.legend(fontsize=7)
plt.suptitle("Node–Allocation MCMC: Source Allocation Posteriors", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "alloc_src_posteriors.png"), dpi=150)
plt.close()

# L2→L3 allocation
fig, axes = plt.subplots(1, N_L2, figsize=(15, 5))
for m in range(N_L2):
    ax = axes[m]
    vname = f"w_l2_det[{m}]"
    if vname in idata.posterior:
        w_post = idata.posterior[vname].stack(
            sample=("chain", "draw")).to_numpy()
        positions = np.arange(N_L3)
        w_true_m = np.array([x_true[idx_23(m, d)] for d in range(N_L3)])
        w_true_m = w_true_m / w_true_m.sum()
        bp = ax.boxplot([w_post[d] for d in range(N_L3)],
                        positions=positions, widths=0.6,
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightyellow")
        ax.scatter(positions, w_true_m, color="red", zorder=5,
                   s=50, marker="D", label="True")
    ax.set_title(f"M{m+1} → L3", fontsize=11)
    ax.set_xticks(range(N_L3))
    ax.set_xticklabels([f"D{d+1}" for d in range(N_L3)], fontsize=8)
    ax.set_ylim(0, 0.6)
    if m == 0:
        ax.legend(fontsize=8)
plt.suptitle("Node–Allocation MCMC: L2 Allocation Posteriors", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "alloc_l2_posteriors.png"), dpi=150)
plt.close()

print(f"\nNode–allocation MCMC complete. Outputs in {OUT_DIR}/")