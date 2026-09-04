#!/usr/bin/env python3
"""
case_study_flow_mcmc.py
Flow-based Bayesian inference for the three-layer case study.

Latent variables: x_f for each of the 45 arcs.
Likelihoods:
  - Individual flow observations: y_f ~ N(x_f, σ_f²)
  - Aggregate throughput: Σ x_L1→L2 ~ N(T_L2_obs, σ_T2²)
  - Aggregate throughput: Σ x_L2→L3 ~ N(T_L3_obs, σ_T3²)
  - Allocation shares (as soft Gaussian penalties on derived shares)
  - Soft mass balance at L2 nodes
  - Soft inequality constraints (via steep penalty)
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
OUT_DIR = "case_study_results/flow_mcmc"
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

# Replicate table: one row per observed arc
rep_idx = np.where(is_obs)[0]
rep_y = obs_values[rep_idx]
rep_sigma = obs_sigma[rep_idx]

# ── Mass balance matrix ─────────────────────────────────────────────
# For each L2 node m: \Sigma_k x_{S_k→M_m} * yield - \Sigma_d x_{M_m→D_d} ≈ 0
# Use nominal yield = 0.95 (uncertain)
nominal_yield = 0.95
n_mb = N_L2
A_mb = np.zeros((n_mb, N_ARCS))
for m in range(N_L2):
    for k in range(N_L1):
        A_mb[m, idx_12(k, m)] = nominal_yield  # inflow * yield
    for d in range(N_L3):
        A_mb[m, idx_23(m, d)] = -1.0  # outflow

SIGMA_BAL = 5.0  # allow some slack for yield uncertainty

# ── Inequality constraint indices ───────────────────────────────────
# We encode inequalities as soft penalties in the model

# ── PyMC model ──────────────────────────────────────────────────────
def main():
    print("Building flow-based PyMC model...")

    # Compute typical scale
    typical_scale = float(np.nanmedian(np.abs(obs_values[is_obs])))
    typical_scale = max(typical_scale, 1.0)

    # Prior centres and widths
    mu0 = np.where(is_obs, obs_values,
                   np.where(np.arange(N_ARCS) < N_ARCS_12,
                            T_L2_obs / N_ARCS_12,
                            T_L3_obs / N_ARCS_23))
        sigma0 = np.where(is_obs, obs_sigma,
                      np.where(np.arange(N_ARCS) < N_ARCS_12,
                               0.8 * T_L2_obs / N_ARCS_12,
                               0.8 * T_L3_obs / N_ARCS_23))

    with pm.Model() as flow_model:
        # ── Priors for each flow ──
        x_list = []
        for f in range(N_ARCS):
            if is_obs[f]:
                xf = pm.TruncatedNormal(
                    f"x[{f}]",
                    mu=float(mu0[f]),
                    sigma=float(sigma0[f]),
                    lower=max(0.2 * mu0[f], 0.01),
                    upper=2.0 * mu0[f],
                )
            else:
                xf = pm.TruncatedNormal(
                    f"x[{f}]",
                    mu=float(mu0[f]),
                    sigma=float(sigma0[f]),
                    lower=0.01,
                )
            x_list.append(xf)

        x = pt.stack(x_list)
        pm.Deterministic("x", x)

        # ── Replicate likelihood ──
        pm.Normal(
            "y_obs",
            mu=x[rep_idx],
            sigma=pt.as_tensor_variable(rep_sigma),
            observed=rep_y,
        )

        # ── Aggregate throughput likelihoods ──
        total_L2 = pt.sum(x[:N_ARCS_12])
        pm.Normal("T_L2_like", mu=total_L2, sigma=sigma_T2,
                  observed=np.array(T_L2_obs))

        total_L3 = pt.sum(x[N_ARCS_12:])
        pm.Normal("T_L3_like", mu=total_L3, sigma=sigma_T3,
                  observed=np.array(T_L3_obs))

        # ── Allocation share likelihoods ──
        # L2 shares: inflow to each M_m / total
        inflow_L2 = pt.stack([
            pt.sum(pt.stack([x[idx_12(k, m)] for k in range(N_L1)]))
            for m in range(N_L2)
        ])
        w_L2_model = inflow_L2 / pt.sum(inflow_L2)
        # Gaussian approximation to Dirichlet likelihood
        sigma_w_L2 = 0.05  # ~5% uncertainty on shares
        pm.Normal("w_L2_like", mu=w_L2_model, sigma=sigma_w_L2,
                  observed=w_L2_obs)

        # L3 shares
        inflow_L3 = pt.stack([
            pt.sum(pt.stack([x[idx_23(m, d)] for m in range(N_L2)]))
            for d in range(N_L3)
        ])
        w_L3_model = inflow_L3 / pt.sum(inflow_L3)
        sigma_w_L3 = 0.06
        pm.Normal("w_L3_like", mu=w_L3_model, sigma=sigma_w_L3,
                  observed=w_L3_obs)

        # ── Soft mass balance ──
        # Mass balance: for each L2 node, inflow*yield - outflow ≈ 0
        # We use a potential (log-likelihood) instead of observed data
        Ax = pt.dot(pt.as_tensor_variable(A_mb), x)
        mb_residuals = Ax / SIGMA_BAL  # standardized residuals
        mb_potential = -0.5 * pt.sum(mb_residuals**2)
        pm.Potential("mass_balance_potential", mb_potential)

        # ── Soft inequality constraints ──
        # Use potentials for inequality constraints
        # S1→M1 ≤ 0.30 * total_L2
        slack_S1M1 = 0.30 * total_L2 - x[idx_12(0, 0)]
        violation_S1M1 = pt.maximum(-slack_S1M1, 0.0) / 5.0
        pm.Potential("ineq_S1M1_potential", -0.5 * violation_S1M1**2)

        # M2 share ≥ 0.20
        slack_M2 = inflow_L2[1] - 0.20 * total_L2
        violation_M2 = pt.maximum(-slack_M2, 0.0) / 5.0
        pm.Potential("ineq_M2_potential", -0.5 * violation_M2**2)

        # D1 inflow ≤ 300
        slack_D1 = 300.0 - inflow_L3[0]
        violation_D1 = pt.maximum(-slack_D1, 0.0) / 5.0
        pm.Potential("ineq_D1_potential", -0.5 * violation_D1**2)

        # Yield constraints: outflow ≤ 0.98 * inflow at each L2 node
        for m in range(N_L2):
            outflow_m = pt.sum(pt.stack([x[idx_23(m, d)] for d in range(N_L3)]))
            inflow_m = pt.sum(pt.stack([x[idx_12(k, m)] for k in range(N_L1)]))
            slack_yield = 0.98 * inflow_m - outflow_m
            violation_yield = pt.maximum(-slack_yield, 0.0) / 2.0
            pm.Potential(f"ineq_yield_M{m+1}_potential", -0.5 * violation_yield**2)

        # ── Sample ──
        print("Sampling (flow-based MCMC)...")
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.92,
            random_seed=42,
            progressbar=True,
        )

    idata = az.from_pymc(trace, model=flow_model)

    # ── Extract results ─────────────────────────────────────────────────
    post_x = idata.posterior["x"]
    x_mean = post_x.mean(dim=("chain", "draw")).to_numpy()
    x_std = post_x.std(dim=("chain", "draw")).to_numpy()
    x_q025 = post_x.quantile(0.025, dim=("chain", "draw")).to_numpy()
    x_q975 = post_x.quantile(0.975, dim=("chain", "draw")).to_numpy()

    x_true = df["true_value"].to_numpy(dtype=float)

    # ── Aggregates ──────────────────────────────────────────────────────
    # Compute per-draw aggregates for uncertainty propagation
    x_samples = post_x.stack(sample=("chain", "draw")).to_numpy()  # (45, n_samples)

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

    # Coverage: fraction of true values within 95% CI
    coverage = np.mean((x_true >= x_q025) & (x_true <= x_q975))

    print(f"\n{'='*60}")
    print(f"FLOW-BASED MCMC RESULTS")
    print(f"{'='*60}")
    print(f"Total L2: true={truth['T_L2_true']:.1f}, "
          f"mean={np.mean(total_L2_samples):.1f} "
          f"[{np.quantile(total_L2_samples, 0.025):.1f}, "
          f"{np.quantile(total_L2_samples, 0.975):.1f}]")
    print(f"Total L3: true={truth['T_L3_true']:.1f}, "
          f"mean={np.mean(total_L3_samples):.1f} "
          f"[{np.quantile(total_L3_samples, 0.025):.1f}, "
          f"{np.quantile(total_L3_samples, 0.975):.1f}]")

    print(f"\nL2 allocation (mean [95% CI]):")
    w_L2_true = np.array(truth["w_L2_agg_true"])
    for m in range(N_L2):
        print(f"  M{m+1}: true={w_L2_true[m]:.3f}, "
              f"mean={np.mean(w_L2_samples[m]):.3f} "
              f"[{np.quantile(w_L2_samples[m], 0.025):.3f}, "
              f"{np.quantile(w_L2_samples[m], 0.975):.3f}]")

    print(f"\nL3 allocation (mean [95% CI]):")
    w_L3_true = np.array(truth["w_L3_agg_true"])
    for d in range(N_L3):
        print(f"  D{d+1}: true={w_L3_true[d]:.3f}, "
              f"mean={np.mean(w_L3_samples[d]):.3f} "
              f"[{np.quantile(w_L3_samples[d], 0.025):.3f}, "
              f"{np.quantile(w_L3_samples[d], 0.975):.3f}]")

    print(f"\nRMSE (all flows): {rmse:.2f} kt/y")
    print(f"MAPE (all flows): {mape:.1f}%")
    print(f"95% CI coverage: {coverage:.1%}")

    # Mass balance residuals at posterior mean
    mb_resid = A_mb @ x_mean
    print(f"Mass balance residuals: max|r|={np.max(np.abs(mb_resid)):.4f}")

    # ── Convergence diagnostics ─────────────────────────────────────────
    summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
    rhat_max = summ["r_hat"].max()
    ess_min = summ["ess_bulk"].min()
    print(f"\nConvergence: max R-hat={rhat_max:.4f}, min ESS={ess_min:.0f}")

    # ── Save ────────────────────────────────────────────────────────────
    df_out = df.copy()
    df_out["posterior_mean"] = x_mean
    df_out["posterior_std"] = x_std
    df_out["ci_lower"] = x_q025
    df_out["ci_upper"] = x_q975
    df_out["error"] = x_mean - x_true
    df_out["in_ci"] = (x_true >= x_q025) & (x_true <= x_q975)
    df_out.to_csv(os.path.join(OUT_DIR, "flow_mcmc_results.csv"), index=False)

    az.to_netcdf(idata, os.path.join(OUT_DIR, "flow_mcmc_trace.nc"))
    summ.to_csv(os.path.join(OUT_DIR, "flow_mcmc_summary.csv"))

    # ── Posterior histograms (selected flows) ───────────────────────────
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    for idx, ax in enumerate(axes.flat):
        if idx >= N_ARCS:
            ax.set_visible(False)
            continue
        vals = x_samples[idx]
        ax.hist(vals, bins=40, density=True, alpha=0.6, color="steelblue")
        ax.axvline(x_true[idx], color="red", ls="--", lw=1.5, label="True")
        ax.axvline(x_mean[idx], color="black", ls="-", lw=1, label="Mean")
        if is_obs[idx]:
            ax.axvline(obs_values[idx], color="green", ls=":", lw=1, label="Obs")
        ax.set_title(f"x[{idx}]", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    plt.suptitle("Flow-Based MCMC: Selected Posterior Marginals", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "flow_mcmc_histograms.png"), dpi=150)
    plt.close()

    print(f"\nFlow-based MCMC complete. Outputs in {OUT_DIR}/")

if __name__ == '__main__':
    main()