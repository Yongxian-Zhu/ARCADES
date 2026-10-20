#!/usr/bin/env python3
"""
case_study_rss_vs_indep.py

Compare RSS-combined vs Independent-Measurement pedigree uncertainty
in a Bayesian Material Flow Analysis of a simplified steel network.

Nodes (6):
    1: Iron Ore (source)
    2: Scrap (source)
    3: Blast Furnace (BF)
    4: Direct Reduction (DR)
    5: Basic Oxygen Furnace (BOF)
    6: Electric Arc Furnace (EAF)

Flows (12):
    f0:  Iron Ore  → BF        = 120 Mt
    f1:  Iron Ore  → DR        =  30 Mt
    f2:  BF        → BOF       =  80 Mt  (hot metal)
    f3:  DR        → EAF       =  28 Mt  (DRI)
    f4:  Scrap     → BOF       =  15 Mt
    f5:  Scrap     → EAF       =  40 Mt
    f6:  BOF       → Steel Out =  90 Mt
    f7:  EAF       → Steel Out =  65 Mt
    f8:  BF        → BF Losses =  40 Mt  (slag + off-gas)
    f9:  DR        → DR Losses =   2 Mt
    f10: BOF       → BOF Losses=   5 Mt
    f11: EAF       → EAF Losses=   3 Mt

Mass balance at internal nodes:
    BF:  f0 = f2 + f8           → 120 = 80 + 40   ✓
    DR:  f1 = f3 + f9           →  30 = 28 +  2   ✓
    BOF: f2 + f4 = f6 + f10     →  80+15 = 90+5   ✓
    EAF: f3 + f5 = f7 + f11     →  28+40 = 65+3   ✓

We deliberately perturb the "observed" values away from perfect balance
to test how each pedigree approach reconciles them.
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  PEDIGREE MODULE  (both approaches, self-contained)
# ═══════════════════════════════════════════════════════════════════════════

DIM_PARAMS = {
    "coverage":         {"r_min": 0.05, "r_max": 0.40},
    "frequency":        {"r_min": 0.05, "r_max": 0.35},
    "spatial_boundary": {"r_min": 0.05, "r_max": 0.30},
}
EPSILON = 1e-6
SCORE_BEST, SCORE_WORST = 1, 4
DIMENSIONS = list(DIM_PARAMS.keys())


def _quality_index(score):
    q = (SCORE_WORST - score) / (SCORE_WORST - SCORE_BEST)
    return float(np.clip(q, 0.0, 1.0))


def _relative_half_range(q_d, dim):
    p = DIM_PARAMS[dim]
    return p["r_max"] - q_d * (p["r_max"] - p["r_min"])


def _dim_sigma(q_d, dim, y):
    return _relative_half_range(q_d, dim) * max(abs(y), EPSILON)


def compute_sigma_rss(scores, y):
    """
    RSS approach: sigma = |y| * sqrt( sum_d r_d^2 )
    Each dimension is an independent ERROR SOURCE on one measurement.
    """
    ss = 0.0
    for dim in DIMENSIONS:
        q = _quality_index(scores[dim])
        r = _relative_half_range(q, dim)
        ss += r ** 2
    return max(abs(y), EPSILON) * np.sqrt(ss)


def compute_sigmas_independent(scores, y):
    """
    Independent-measurement approach: each dimension yields its own sigma.
    Returns dict {dim: sigma_d} plus the effective combined sigma from
    precision-sum:  sigma_eff = 1/sqrt(sum_d 1/sigma_d^2)
    """
    sigmas = {}
    tau_sum = 0.0
    for dim in DIMENSIONS:
        q = _quality_index(scores[dim])
        s = _dim_sigma(q, dim, y)
        sigmas[dim] = s
        tau_sum += 1.0 / (s ** 2)
    sigmas["effective"] = 1.0 / np.sqrt(tau_sum) if tau_sum > 0 else np.inf
    return sigmas


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC NETWORK DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

# Node numbering
NODES = {
    1: "Iron Ore",
    2: "Scrap",
    3: "Blast Furnace",
    4: "Direct Reduction",
    5: "Basic Oxygen Furnace",
    6: "Electric Arc Furnace",
    7: "Steel Out",       # sink
    8: "BF Losses",       # sink
    9: "DR Losses",       # sink
    10: "BOF Losses",     # sink
    11: "EAF Losses",     # sink
}

# True flows (Mt/yr) — these satisfy mass balance exactly
TRUE_FLOWS = np.array([
    120.0,  # f0:  IronOre → BF
     30.0,  # f1:  IronOre → DR
     80.0,  # f2:  BF      → BOF
     28.0,  # f3:  DR      → EAF
     15.0,  # f4:  Scrap   → BOF
     40.0,  # f5:  Scrap   → EAF
     90.0,  # f6:  BOF     → SteelOut
     65.0,  # f7:  EAF     → SteelOut
     40.0,  # f8:  BF      → BF_Losses
      2.0,  # f9:  DR      → DR_Losses
      5.0,  # f10: BOF     → BOF_Losses
      3.0,  # f11: EAF     → EAF_Losses
])

FLOW_META = pd.DataFrame({
    "flow_idx":         range(12),
    "from_node_number": [1, 1, 3, 4, 2, 2, 5, 6, 3, 4, 5, 6],
    "to_node_number":   [3, 4, 5, 6, 5, 6, 7, 7, 8, 9, 10, 11],
    "from_name": [
        "Iron Ore", "Iron Ore", "BF", "DR", "Scrap", "Scrap",
        "BOF", "EAF", "BF", "DR", "BOF", "EAF",
    ],
    "to_name": [
        "BF", "DR", "BOF", "EAF", "BOF", "EAF",
        "Steel Out", "Steel Out", "BF Losses", "DR Losses",
        "BOF Losses", "EAF Losses",
    ],
    "true_value": TRUE_FLOWS,
})

# Pedigree scores (1=best, 4=worst) — deliberately varied
# Flows with poor scores will have large uncertainty
PEDIGREE_SCORES = pd.DataFrame({
    "flow_idx": range(12),
    #                       cov  freq  spat
    "coverage":            [ 1,   2,   1,   3,   2,   1,   1,   2,   3,   4,   2,   3],
    "frequency":           [ 1,   2,   2,   3,   1,   2,   1,   3,   2,   4,   3,   3],
    "spatial_boundary":    [ 1,   1,   2,   2,   2,   1,   1,   2,   3,   3,   2,   4],
})

# Observed values: true + noise + deliberate bias to break mass balance
np.random.seed(42)
OBSERVED_NOISE_FRAC = 0.08  # 8% Gaussian noise on true values
OBSERVED_VALUES = TRUE_FLOWS * (1.0 + OBSERVED_NOISE_FRAC * np.random.randn(12))
# Add extra bias to a few flows to stress-test reconciliation
OBSERVED_VALUES[0] += 5.0   # Iron Ore → BF observed too high
OBSERVED_VALUES[6] -= 3.0   # BOF → Steel observed too low
OBSERVED_VALUES[11] += 1.5  # EAF losses observed too high

N_FLOWS = len(TRUE_FLOWS)

# ═══════════════════════════════════════════════════════════════════════════
# 3.  MASS-BALANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

# Internal nodes: BF(3), DR(4), BOF(5), EAF(6)
# For each: sum(inflows) - sum(outflows) = 0

def build_mass_balance_matrix():
    """
    A_bal @ x = 0  for internal nodes.
    Rows: one per internal node.
    Columns: one per flow.
    +1 for inflows, -1 for outflows.
    """
    internal_nodes = [3, 4, 5, 6]
    from_nodes = FLOW_META["from_node_number"].values
    to_nodes = FLOW_META["to_node_number"].values

    A = np.zeros((len(internal_nodes), N_FLOWS))
    for row_i, node in enumerate(internal_nodes):
        for j in range(N_FLOWS):
            if to_nodes[j] == node:
                A[row_i, j] = +1.0   # inflow
            if from_nodes[j] == node:
                A[row_i, j] = -1.0   # outflow
    return A, internal_nodes

A_BAL, MB_NODES = build_mass_balance_matrix()

# Verify on true flows
assert np.allclose(A_BAL @ TRUE_FLOWS, 0.0), "True flows don't satisfy mass balance!"
print("Mass-balance matrix A (4 constraints × 12 flows):")
print(A_BAL)
print(f"A @ true_flows = {A_BAL @ TRUE_FLOWS}")
print(f"A @ observed   = {A_BAL @ OBSERVED_VALUES}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 4.  COMPUTE PEDIGREE SIGMAS UNDER BOTH APPROACHES
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_sigmas():
    """Return DataFrames with per-flow sigma under RSS and independent approaches."""
    records = []
    for j in range(N_FLOWS):
        scores = {d: PEDIGREE_SCORES.loc[j, d] for d in DIMENSIONS}
        y = OBSERVED_VALUES[j]

        sigma_rss = compute_sigma_rss(scores, y)
        sigmas_indep = compute_sigmas_independent(scores, y)

        rec = {
            "flow_idx": j,
            "from": FLOW_META.loc[j, "from_name"],
            "to": FLOW_META.loc[j, "to_name"],
            "true_value": TRUE_FLOWS[j],
            "observed": y,
            "coverage": scores["coverage"],
            "frequency": scores["frequency"],
            "spatial_boundary": scores["spatial_boundary"],
            "sigma_rss": sigma_rss,
            "sigma_coverage": sigmas_indep["coverage"],
            "sigma_frequency": sigmas_indep["frequency"],
            "sigma_spatial": sigmas_indep["spatial_boundary"],
            "sigma_indep_eff": sigmas_indep["effective"],
        }
        records.append(rec)

    return pd.DataFrame(records)


df_sigmas = compute_all_sigmas()
print("=" * 90)
print("PEDIGREE SIGMA COMPARISON")
print("=" * 90)
print(df_sigmas[[
    "flow_idx", "from", "to", "observed",
    "coverage", "frequency", "spatial_boundary",
    "sigma_rss", "sigma_indep_eff",
]].to_string(index=False))
print()

# Show the ratio
df_sigmas["ratio_rss_over_indep"] = df_sigmas["sigma_rss"] / df_sigmas["sigma_indep_eff"]
print("sigma_rss / sigma_indep_eff  (always > 1 for D>1 dimensions):")
print(df_sigmas[["flow_idx", "from", "to", "ratio_rss_over_indep"]].to_string(index=False))
print()


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BAYESIAN MFA MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════

SIGMA_BALANCE = 1e-3   # soft mass-balance tolerance
MIN_POSITIVE = 1e-6
N_CHAINS = 4
N_DRAWS = 2000
N_TUNE = 2000
SEED = 123


def build_and_sample_rss(tag="RSS"):
    """
    Standard approach: one likelihood term per flow, sigma from RSS pedigree.
    y_j ~ Normal(x_j, sigma_rss_j)
    """
    sigma_vec = df_sigmas["sigma_rss"].values

    with pm.Model() as model:
        x_list = []
        for j in range(N_FLOWS):
            mu0 = float(OBSERVED_VALUES[j])
            sd0 = float(max(sigma_vec[j], 1e-3))
            lb = max(0.2 * mu0, MIN_POSITIVE) if mu0 > 0 else MIN_POSITIVE
            ub = 1.8 * mu0 if mu0 > 0 else 500.0
            xj = pm.TruncatedNormal(
                f"x_{j}", mu=mu0, sigma=sd0,
                lower=lb, upper=ub,
            )
            x_list.append(xj)

        x = pt.stack(x_list)
        pm.Deterministic("x", x)

        # Single likelihood per flow
        pm.Normal(
            "y_obs",
            mu=x,
            sigma=pt.as_tensor_variable(sigma_vec),
            observed=OBSERVED_VALUES,
        )

                # Soft mass balance
        Ax = pt.dot(pt.as_tensor_variable(A_BAL), x)
        pm.Normal(
            "mass_balance",
            mu=Ax,
            sigma=SIGMA_BALANCE,
            observed=np.zeros(len(MB_NODES)),
        )

        trace = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
            cores=min(4, N_CHAINS), target_accept=0.95,
            random_seed=SEED, progressbar=True,
        )

    idata = az.from_pymc(trace, model=model)
    print(f"\n[{tag}] Sampling complete.")
    return idata


def build_and_sample_independent(tag="INDEP"):
    """
    Independent-measurement approach: D=3 likelihood terms per flow.
    For flow j and dimension d:
        y_j ~ Normal(x_j, sigma_{j,d})

    Each flow contributes 3 likelihood factors, each with its own sigma.
    """
    # Build long-format observation arrays
    obs_y_long = []      # repeated observed value
    obs_sigma_long = []  # per-dimension sigma
    obs_flow_idx = []    # which latent x this observation constrains

    for j in range(N_FLOWS):
        scores = {d: PEDIGREE_SCORES.loc[j, d] for d in DIMENSIONS}
        y = OBSERVED_VALUES[j]
        sigmas = compute_sigmas_independent(scores, y)

        for dim in DIMENSIONS:
            obs_y_long.append(y)
            obs_sigma_long.append(sigmas[dim])
            obs_flow_idx.append(j)

    obs_y_long = np.array(obs_y_long)
    obs_sigma_long = np.array(obs_sigma_long)
    obs_flow_idx = np.array(obs_flow_idx, dtype=int)

    n_obs = len(obs_y_long)
    print(f"[{tag}] Total likelihood terms: {n_obs} "
          f"({N_FLOWS} flows × {len(DIMENSIONS)} dimensions)")

    with pm.Model() as model:
        x_list = []
        for j in range(N_FLOWS):
            mu0 = float(OBSERVED_VALUES[j])
            # Use the effective (precision-sum) sigma for the prior width
            sd0 = float(max(df_sigmas.loc[j, "sigma_rss"], 1e-3))
            lb = max(0.2 * mu0, MIN_POSITIVE) if mu0 > 0 else MIN_POSITIVE
            ub = 1.8 * mu0 if mu0 > 0 else 500.0
            xj = pm.TruncatedNormal(
                f"x_{j}", mu=mu0, sigma=sd0,
                lower=lb, upper=ub,
            )
            x_list.append(xj)

        x = pt.stack(x_list)
        pm.Deterministic("x", x)

        # D likelihood terms per flow (the key difference)
        pm.Normal(
            "y_obs_dim",
            mu=x[obs_flow_idx],
            sigma=pt.as_tensor_variable(obs_sigma_long),
            observed=obs_y_long,
        )

                # Soft mass balance (identical to RSS version)
        Ax = pt.dot(pt.as_tensor_variable(A_BAL), x)
        pm.Normal(
            "mass_balance",
            mu=Ax,
            sigma=SIGMA_BALANCE,
            observed=np.zeros(len(MB_NODES)),
        )

        trace = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
            cores=min(4, N_CHAINS), target_accept=0.95,
            random_seed=SEED, progressbar=True,
        )

    idata = az.from_pymc(trace, model=model)
    print(f"\n[{tag}] Sampling complete.")
    return idata


# ═══════════════════════════════════════════════════════════════════════════
# 6.  RUN BOTH MODELS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 90)
    print("RUNNING RSS MODEL")
    print("=" * 90)
    idata_rss = build_and_sample_rss("RSS")

    print("\n" + "=" * 90)
    print("RUNNING INDEPENDENT-MEASUREMENT MODEL")
    print("=" * 90)
    idata_indep = build_and_sample_independent("INDEP")


    # ═══════════════════════════════════════════════════════════════════════════
    # 7.  EXTRACT POSTERIORS AND COMPARE
    # ═══════════════════════════════════════════════════════════════════════════

def extract_posterior_stats(idata, label):
    """Extract mean, std, HDI for each flow from an InferenceData object."""
    post = idata.posterior["x"]
    # Shape: (chain, draw, flow) or (chain, draw) with flow as last dim
    flat = post.values.reshape(-1, N_FLOWS)  # (n_samples, n_flows)

    records = []
    for j in range(N_FLOWS):
        samples = flat[:, j]
        lo, hi = az.hdi(samples, hdi_prob=0.95)
        records.append({
            "flow_idx": j,
            "from": FLOW_META.loc[j, "from_name"],
            "to": FLOW_META.loc[j, "to_name"],
            "true": TRUE_FLOWS[j],
            "observed": OBSERVED_VALUES[j],
            f"mean_{label}": np.mean(samples),
            f"std_{label}": np.std(samples),
            f"hdi_lo_{label}": lo,
            f"hdi_hi_{label}": hi,
            f"hdi_width_{label}": hi - lo,
            f"error_{label}": np.mean(samples) - TRUE_FLOWS[j],
            f"abs_error_{label}": abs(np.mean(samples) - TRUE_FLOWS[j]),
        })
    return pd.DataFrame(records)


    df_rss = extract_posterior_stats(idata_rss, "rss")
    df_indep = extract_posterior_stats(idata_indep, "indep")

    # Merge
    df_compare = df_rss.merge(
        df_indep.drop(columns=["from", "to", "true", "observed"]),
        on="flow_idx",
    )

    # Derived comparison columns
    df_compare["std_ratio"] = df_compare["std_rss"] / df_compare["std_indep"]
    df_compare["hdi_ratio"] = df_compare["hdi_width_rss"] / df_compare["hdi_width_indep"]
    df_compare["closer_to_true"] = np.where(
        df_compare["abs_error_rss"] < df_compare["abs_error_indep"],
        "RSS", "INDEP"
    )

    # Mass-balance residuals
    post_rss_mean = df_compare["mean_rss"].values
    post_indep_mean = df_compare["mean_indep"].values
    mb_resid_rss = A_BAL @ post_rss_mean
    mb_resid_indep = A_BAL @ post_indep_mean

    print("\n" + "=" * 90)
    print("POSTERIOR COMPARISON TABLE")
    print("=" * 90)
    display_cols = [
        "flow_idx", "from", "to", "true", "observed",
        "mean_rss", "std_rss", "hdi_width_rss",
        "mean_indep", "std_indep", "hdi_width_indep",
        "std_ratio", "closer_to_true",
    ]
    print(df_compare[display_cols].to_string(index=False, float_format="%.3f"))

    print("\n" + "-" * 60)
    print("SUMMARY STATISTICS")
    print("-" * 60)
    print(f"  Mean posterior std  (RSS):   {df_compare['std_rss'].mean():.4f}")
    print(f"  Mean posterior std  (INDEP): {df_compare['std_indep'].mean():.4f}")
    print(f"  Mean HDI width     (RSS):   {df_compare['hdi_width_rss'].mean():.4f}")
    print(f"  Mean HDI width     (INDEP): {df_compare['hdi_width_indep'].mean():.4f}")
    print(f"  Mean |error|       (RSS):   {df_compare['abs_error_rss'].mean():.4f}")
    print(f"  Mean |error|       (INDEP): {df_compare['abs_error_indep'].mean():.4f}")
    print(f"  Std ratio (RSS/INDEP) mean: {df_compare['std_ratio'].mean():.3f}")
    print()
    print(f"  MB residual L2 (RSS):   {np.linalg.norm(mb_resid_rss):.6f}")
    print(f"  MB residual L2 (INDEP): {np.linalg.norm(mb_resid_indep):.6f}")
    print()
    print(f"  Flows closer to true (RSS):   "
          f"{(df_compare['closer_to_true'] == 'RSS').sum()}/{N_FLOWS}")
    print(f"  Flows closer to true (INDEP): "
          f"{(df_compare['closer_to_true'] == 'INDEP').sum()}/{N_FLOWS}")


# ═══════════════════════════════════════════════════════════════════════════
    # 8.  VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    OUT_DIR = "case_study_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save comparison table
    df_compare.to_csv(os.path.join(OUT_DIR, "posterior_comparison.csv"), index=False)
    df_sigmas.to_csv(os.path.join(OUT_DIR, "pedigree_sigmas.csv"), index=False)

    # ---------- Figure 1: Pedigree sigma comparison ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(N_FLOWS)
    width = 0.35
    ax.bar(x_pos - width / 2, df_sigmas["sigma_rss"], width,
           label="σ RSS", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width / 2, df_sigmas["sigma_indep_eff"], width,
           label="σ Indep (effective)", color="coral", alpha=0.8)
    ax.set_xlabel("Flow index")
    ax.set_ylabel("σ (Mt)")
    ax.set_title("Pedigree Uncertainty: RSS vs Independent-Measurement")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{j}" for j in range(N_FLOWS)], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_sigma_comparison.png"), dpi=150)
    plt.close(fig)

    # ---------- Figure 2: Posterior means vs truth ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2
    ax.bar(x_pos - 1.5 * width, TRUE_FLOWS, width,
           label="True", color="green", alpha=0.7)
    ax.bar(x_pos - 0.5 * width, OBSERVED_VALUES, width,
           label="Observed", color="gray", alpha=0.5)
    ax.bar(x_pos + 0.5 * width, df_compare["mean_rss"], width,
           label="Posterior (RSS)", color="steelblue", alpha=0.8,
           yerr=df_compare["std_rss"], capsize=3)
    ax.bar(x_pos + 1.5 * width, df_compare["mean_indep"], width,
           label="Posterior (Indep)", color="coral", alpha=0.8,
           yerr=df_compare["std_indep"], capsize=3)
    ax.set_xlabel("Flow index")
    ax.set_ylabel("Flow (Mt)")
    ax.set_title("Posterior Means vs True Values")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{j}\n{FLOW_META.loc[j,'from_name'][:3]}→{FLOW_META.loc[j,'to_name'][:3]}"
                         for j in range(N_FLOWS)], fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_posterior_means.png"), dpi=150)
    plt.close(fig)

    # ---------- Figure 3: Posterior histograms side by side ----------
    post_rss_flat = idata_rss.posterior["x"].values.reshape(-1, N_FLOWS)
    post_indep_flat = idata_indep.posterior["x"].values.reshape(-1, N_FLOWS)

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.3)

    for j in range(N_FLOWS):
        ax = fig.add_subplot(gs[j // 3, j % 3])

        ax.hist(post_rss_flat[:, j], bins=60, density=True, alpha=0.5,
                color="steelblue", label="RSS")
        ax.hist(post_indep_flat[:, j], bins=60, density=True, alpha=0.5,
                color="coral", label="Indep")
        ax.axvline(TRUE_FLOWS[j], color="green", lw=2, ls="--", label="True")
        ax.axvline(OBSERVED_VALUES[j], color="gray", lw=1.5, ls=":", label="Obs")

        ax.set_title(
            f"f{j}: {FLOW_META.loc[j,'from_name']}→{FLOW_META.loc[j,'to_name']}",
            fontsize=9,
        )
        ax.set_xlabel("Mt", fontsize=8)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Posterior Distributions: RSS (blue) vs Independent (red)", fontsize=14)
    fig.savefig(os.path.join(OUT_DIR, "fig3_posterior_histograms.png"), dpi=150)
    plt.close(fig)

    # ---------- Figure 4: HDI width comparison ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_pos - width / 2, df_compare["hdi_width_rss"], width,
           label="95% HDI width (RSS)", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width / 2, df_compare["hdi_width_indep"], width,
           label="95% HDI width (Indep)", color="coral", alpha=0.8)
    ax.set_xlabel("Flow index")
    ax.set_ylabel("HDI width (Mt)")
    ax.set_title("95% HDI Width: RSS vs Independent-Measurement")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{j}" for j in range(N_FLOWS)], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_hdi_width.png"), dpi=150)
    plt.close(fig)

    # ---------- Figure 5: Std ratio and error comparison ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = ["steelblue" if r > 1 else "coral" for r in df_compare["std_ratio"]]
    ax.bar(x_pos, df_compare["std_ratio"], color=colors, alpha=0.8)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("Flow index")
    ax.set_ylabel("std(RSS) / std(Indep)")
    ax.set_title("Posterior Std Ratio\n(>1 means RSS is wider)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{j}" for j in range(N_FLOWS)])
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x_pos - width / 2, df_compare["abs_error_rss"], width,
           label="|error| RSS", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width / 2, df_compare["abs_error_indep"], width,
           label="|error| Indep", color="coral", alpha=0.8)
    ax.set_xlabel("Flow index")
    ax.set_ylabel("|Posterior mean − True| (Mt)")
    ax.set_title("Absolute Error vs Truth")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{j}" for j in range(N_FLOWS)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_ratio_and_error.png"), dpi=150)
    plt.close(fig)

    # ---------- Figure 6: Mass-balance residuals ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    mb_x = np.arange(len(MB_NODES))
    ax.bar(mb_x - 0.15, mb_resid_rss, 0.3,
           label="MB residual (RSS)", color="steelblue", alpha=0.8)
    ax.bar(mb_x + 0.15, mb_resid_indep, 0.3,
           label="MB residual (Indep)", color="coral", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Internal node")
    ax.set_ylabel("Residual (Mt)")
    ax.set_title("Mass-Balance Residuals at Posterior Mean")
    ax.set_xticks(mb_x)
    ax.set_xticklabels([NODES[n] for n in MB_NODES], rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_mass_balance_residuals.png"), dpi=150)
    plt.close(fig)

    # ---------- Figure 7: Per-dimension sigma breakdown ----------
    fig, ax = plt.subplots(figsize=(14, 5))
    w = 0.18
    for i, dim in enumerate(DIMENSIONS):
        col = f"sigma_{dim.split('_')[0]}" if dim != "spatial_boundary" else "sigma_spatial"
        ax.bar(x_pos + (i - 1) * w, df_sigmas[col], w,
               label=f"σ_{dim}", alpha=0.8)
    ax.bar(x_pos + 2 * w, df_sigmas["sigma_rss"], w,
           label="σ_RSS", color="steelblue", alpha=0.6, edgecolor="black")
    ax.bar(x_pos + 3 * w, df_sigmas["sigma_indep_eff"], w,
           label="σ_indep_eff", color="coral", alpha=0.6, edgecolor="black")
    ax.set_xlabel("Flow index")
    ax.set_ylabel("σ (Mt)")
    ax.set_title("Per-Dimension Sigmas and Combined Values")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"f{j}" for j in range(N_FLOWS)])
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_per_dim_sigmas.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
    # 9.  DIAGNOSTIC CHECKS
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("MCMC DIAGNOSTICS")
    print("=" * 90)

    for label, idata in [("RSS", idata_rss), ("INDEP", idata_indep)]:
        summ = az.summary(idata, var_names=["x"], hdi_prob=0.95)
        summ.to_csv(os.path.join(OUT_DIR, f"summary_{label.lower()}.csv"))

        rhat_max = summ["r_hat"].max()
        ess_min = summ["ess_bulk"].min()
        print(f"  [{label}]  R-hat max: {rhat_max:.4f}   ESS_bulk min: {ess_min:.0f}")
        if rhat_max > 1.05:
            print(f"    ⚠ WARNING: R-hat > 1.05 detected — chains may not have converged.")
        if ess_min < 400:
            print(f"    ⚠ WARNING: Low ESS — consider more draws or tuning.")


    # ═══════════════════════════════════════════════════════════════════════════
    # 10.  FINAL INTERPRETIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
RSS approach:
  - Each pedigree dimension is an independent ERROR SOURCE.
  - sigma_combined = |y| * sqrt(sum r_d^2)  →  grows with more dimensions.
  - Wider posteriors, more conservative.
  - The model is less confident about each flow.

Independent-measurement approach:
  - Each pedigree dimension is an independent CORROBORATING OBSERVATION.
  - Effective sigma = 1/sqrt(sum 1/sigma_d^2)  →  shrinks with more dimensions.
  - Narrower posteriors, more informative.
  - The model gains confidence when multiple quality dimensions agree.

Key differences in MFA context:
  - The independent approach produces TIGHTER credible intervals.
  - With tight mass-balance constraints, both approaches are pulled toward
    balance, but the independent approach has less tension between the
    likelihood and the balance constraint.
  - For flows with MIXED pedigree scores (e.g., good coverage but poor
    frequency), the independent approach lets the well-scored dimension
    dominate, while RSS inflates uncertainty from the worst dimension.
""")

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print("  - posterior_comparison.csv")
    print("  - pedigree_sigmas.csv")
    print("  - summary_rss.csv / summary_indep.csv")
    print("  - fig1–fig7 (PNG)")