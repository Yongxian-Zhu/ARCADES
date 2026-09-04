#!/usr/bin/env python3
"""
case_study_node_alloc.py

Bayesian MFA using node throughputs and allocation coefficients
as design variables instead of raw flows.

Key advantage: mass balance is satisfied BY CONSTRUCTION.

Nodes (6 internal + sinks):
    1: Iron Ore (source)
    2: Scrap (source)
    3: Blast Furnace (BF)
    4: Direct Reduction (DR)
    5: Basic Oxygen Furnace (BOF)
    6: Electric Arc Furnace (EAF)
    7: Steel Out (sink)
    8+: Loss sinks

Design variables:
    - T_n: throughput at each source/process node (6 values)
    - alpha_{n->m}: allocation fractions at each node (sum to 1)

Flows are deterministic:
    f_{n->m} = T_n * alpha_{n->m}
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  PEDIGREE MODULE (same as before)
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
    return p["r_min"] + (1 - q_d) * (p["r_max"] - p["r_min"])


def compute_sigma_rss(scores, y):
    ss = 0.0
    for dim in DIMENSIONS:
        q = _quality_index(scores[dim])
        r = _relative_half_range(q, dim) / 1.96
        ss += r ** 2
    return max(abs(y), EPSILON) * np.sqrt(ss)


def compute_sigmas_independent(scores, y):
    sigmas = {}
    tau_sum = 0.0
    for dim in DIMENSIONS:
        q = _quality_index(scores[dim])
        s = _relative_half_range(q, dim) * max(abs(y), EPSILON) / 1.96
        sigmas[dim] = s
        tau_sum += 1.0 / (s ** 2)
    sigmas["effective"] = 1.0 / np.sqrt(tau_sum) if tau_sum > 0 else np.inf
    return sigmas


# ═══════════════════════════════════════════════════════════════════════════
# 2.  NETWORK DEFINITION (node-allocation parameterization)
# ═══════════════════════════════════════════════════════════════════════════

# --- Node structure ---
# Each source/process node has a throughput and allocation vector.
# Allocation vectors sum to 1.

# Node definitions with their output destinations
NODE_DEFS = {
    "IronOre": {
        "node_id": 1,
        "type": "source",
        "outputs": ["BF", "DR"],           # where output goes
        "output_node_ids": [3, 4],
    },
    "Scrap": {
        "node_id": 2,
        "type": "source",
        "outputs": ["BOF", "EAF"],
        "output_node_ids": [5, 6],
    },
    "BF": {
        "node_id": 3,
        "type": "process",
        "outputs": ["BOF", "EAF", "BF_Loss"],
        "output_node_ids": [5, 6, 8],
    },
    "DR": {
        "node_id": 4,
        "type": "process",
        "outputs": ["EAF", "DR_Loss"],
        "output_node_ids": [6, 9],
    },
    "BOF": {
        "node_id": 5,
        "type": "process",
        "outputs": ["SteelOut", "BOF_Loss"],
        "output_node_ids": [7, 10],
    },
    "EAF": {
        "node_id": 6,
        "type": "process",
        "outputs": ["SteelOut", "EAF_Loss"],
        "output_node_ids": [7, 11],
    },
}

# --- True values ---
TRUE_THROUGHPUTS = {
    "IronOre": 150.0,
    "Scrap":    55.0,
    "BF":      120.0,
    "DR":       30.0,
    "BOF":      87.0,   # 72 from BF + 15 from Scrap
    "EAF":      76.0,   # 28 from DR + 40 from Scrap + 8 from BF
}

TRUE_ALLOCATIONS = {
    "IronOre": [120.0/150.0, 30.0/150.0],           # [0.8, 0.2]
    "Scrap":   [15.0/55.0, 40.0/55.0],              # [0.2727, 0.7273]
    "BF":      [72.0/120.0, 8.0/120.0, 40.0/120.0], # [0.6, 0.0667, 0.3333]
    "DR":      [28.0/30.0, 2.0/30.0],               # [0.9333, 0.0667]
    "BOF":     [82.0/87.0, 5.0/87.0],               # [0.9425, 0.0575]
    "EAF":     [73.0/76.0, 3.0/76.0],               # [0.9605, 0.0395]
}

# Verify: compute all true flows from throughputs and allocations
def compute_flows(throughputs, allocations):
    """Given throughputs and allocations, compute all flows."""
    flows = {}
    for node_name, node_def in NODE_DEFS.items():
        T = throughputs[node_name]
        alphas = allocations[node_name]
        for i, dest in enumerate(node_def["outputs"]):
            flow_name = f"{node_name}->{dest}"
            flows[flow_name] = T * alphas[i]
    return flows

TRUE_FLOWS = compute_flows(TRUE_THROUGHPUTS, TRUE_ALLOCATIONS)

# Verify mass balance at process nodes
def check_mass_balance(flows):
    """Check that inflows = outflows at each process node."""
    for node_name, node_def in NODE_DEFS.items():
        if node_def["type"] != "process":
            continue
        # Compute total inflow
        inflow = 0.0
        for src_name, src_def in NODE_DEFS.items():
            for i, dest in enumerate(src_def["outputs"]):
                if dest == node_name:
                    flow_name = f"{src_name}->{dest}"
                    inflow += flows[flow_name]
        # Compute total outflow
        outflow = sum(
            flows[f"{node_name}->{dest}"]
            for dest in node_def["outputs"]
        )
        residual = inflow - outflow
        print(f"  {node_name}: in={inflow:.2f}, out={outflow:.2f}, "
              f"residual={residual:.6f}")
    print()

print("True flows:")
for name, val in TRUE_FLOWS.items():
    print(f"  {name}: {val:.4f}")
print()
print("Mass balance check (true):")
check_mass_balance(TRUE_FLOWS)


# --- Build ordered flow list for observations ---
FLOW_NAMES = list(TRUE_FLOWS.keys())
N_FLOWS = len(FLOW_NAMES)
TRUE_FLOW_ARRAY = np.array([TRUE_FLOWS[fn] for fn in FLOW_NAMES])

# --- Pedigree scores per flow ---
# Order matches FLOW_NAMES
PEDIGREE_SCORES_LIST = [
    # IronOre->BF, IronOre->DR, Scrap->BOF, Scrap->EAF,
    # BF->BOF, BF->EAF, BF->BF_Loss,
    # DR->EAF, DR->DR_Loss,
    # BOF->SteelOut, BOF->BOF_Loss,
    # EAF->SteelOut, EAF->EAF_Loss
    {"coverage": 1, "frequency": 1, "spatial_boundary": 1},  # IronOre->BF
    {"coverage": 2, "frequency": 2, "spatial_boundary": 1},  # IronOre->DR
    {"coverage": 2, "frequency": 1, "spatial_boundary": 2},  # Scrap->BOF
    {"coverage": 1, "frequency": 2, "spatial_boundary": 1},  # Scrap->EAF
    {"coverage": 1, "frequency": 2, "spatial_boundary": 2},  # BF->BOF
    {"coverage": 2, "frequency": 2, "spatial_boundary": 2},  # BF->EAF
    {"coverage": 3, "frequency": 2, "spatial_boundary": 3},  # BF->BF_Loss
    {"coverage": 3, "frequency": 3, "spatial_boundary": 2},  # DR->EAF
    {"coverage": 4, "frequency": 4, "spatial_boundary": 3},  # DR->DR_Loss
    {"coverage": 1, "frequency": 1, "spatial_boundary": 1},  # BOF->SteelOut
    {"coverage": 2, "frequency": 3, "spatial_boundary": 2},  # BOF->BOF_Loss
    {"coverage": 2, "frequency": 3, "spatial_boundary": 2},  # EAF->SteelOut
    {"coverage": 3, "frequency": 3, "spatial_boundary": 4},  # EAF->EAF_Loss
]

# --- Generate observed flow values ---
np.random.seed(42)
OBSERVED_NOISE_FRAC = 0.08
OBSERVED_FLOWS = TRUE_FLOW_ARRAY * (
    1.0 + OBSERVED_NOISE_FRAC * np.random.randn(N_FLOWS)
)
# Add deliberate biases
OBSERVED_FLOWS[0] += 5.0   # IronOre->BF too high
OBSERVED_FLOWS[9] -= 3.0   # BOF->SteelOut too low
OBSERVED_FLOWS[12] += 1.5  # EAF->EAF_Loss too high

# --- Compute pedigree sigmas for each flow ---
SIGMA_RSS = np.array([
    compute_sigma_rss(PEDIGREE_SCORES_LIST[j], OBSERVED_FLOWS[j])
    for j in range(N_FLOWS)
])

SIGMAS_INDEP = [
    compute_sigmas_independent(PEDIGREE_SCORES_LIST[j], OBSERVED_FLOWS[j])
    for j in range(N_FLOWS)
]

print("Observed flows and sigmas:")
for j, fn in enumerate(FLOW_NAMES):
    print(f"  {fn:20s}: true={TRUE_FLOW_ARRAY[j]:8.2f}  "
          f"obs={OBSERVED_FLOWS[j]:8.2f}  "
          f"sigma_rss={SIGMA_RSS[j]:6.2f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 3.  RATIO OBSERVATION DATA
# ═══════════════════════════════════════════════════════════════════════════

# "90% of BF product (excl. losses) goes to BOF, 10% to EAF"
# In allocation terms: alpha_{BF->BOF} / (alpha_{BF->BOF} + alpha_{BF->EAF}) = 0.9
# Or equivalently on the non-loss allocations.
# We observe this as a constraint on the BF allocation vector.

RATIO_OBS = {
    "node": "BF",
    "numerator_output": "BOF",     # index 0 in BF outputs
    "denominator_outputs": ["BOF", "EAF"],  # indices 0,1 in BF outputs
    "observed_ratio": 0.90,
    "sigma_ratio": 0.03,           # uncertainty on the ratio
}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  HELPER: map from (throughputs, allocations) to flow vector
# ═══════════════════════════════════════════════════════════════════════════

NODE_NAMES = list(NODE_DEFS.keys())  # ordered list

def get_flow_index(flow_name):
    """Get index of a flow in FLOW_NAMES."""
    return FLOW_NAMES.index(flow_name)


def build_throughput_consistency_matrix():
    """
    For each process node, throughput must equal sum of inflows.
    T_n = sum of all flows into node n.

    This is NOT mass balance (which is automatic).
    This links throughputs to each other via allocations.
    """
    process_nodes = [n for n, d in NODE_DEFS.items() if d["type"] == "process"]
    # For each process node, find which flows feed into it
    inflow_map = {}
    for node_name in process_nodes:
        inflows = []
        for src_name, src_def in NODE_DEFS.items():
            for i, dest in enumerate(src_def["outputs"]):
                if dest == node_name:
                    inflows.append((src_name, i))
        inflow_map[node_name] = inflows
    return inflow_map, process_nodes


INFLOW_MAP, PROCESS_NODES = build_throughput_consistency_matrix()

print("Inflow map (which flows feed each process node):")
for node, inflows in INFLOW_MAP.items():
    parts = [f"{src}[alloc_{i}]" for src, i in inflows]
    print(f"  {node}: {', '.join(parts)}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BAYESIAN MODEL: NODE-ALLOCATION PARAMETERIZATION
# ═══════════════════════════════════════════════════════════════════════════

N_CHAINS = 4
N_DRAWS = 2000
N_TUNE = 2000
SEED = 123


def build_and_sample_node_alloc(approach="RSS", tag="RSS"):
    """
    Build Bayesian model with throughputs and allocations as latent variables.

    Flows are deterministic: f_{n->m} = T_n * alpha_{n->m}
    Mass balance is automatic because allocations sum to 1.

    Observations are on FLOWS (not directly on throughputs/allocations),
    so the likelihood compares observed flows to computed flows.
    """

    with pm.Model() as model:

        # ── Latent throughputs ──
        T = {}
        for node_name in NODE_NAMES:
            true_T = TRUE_THROUGHPUTS[node_name]
            T[node_name] = pm.TruncatedNormal(
                f"T_{node_name}",
                mu=true_T * 1.05,  # slightly biased initial guess
                sigma=true_T * 0.3,
                lower=true_T * 0.2,
                upper=true_T * 2.0,
            )

        # ── Latent allocations (Dirichlet) ──
        alpha = {}
        for node_name in NODE_NAMES:
            n_outputs = len(NODE_DEFS[node_name]["outputs"])
            true_alpha = np.array(TRUE_ALLOCATIONS[node_name])

            # Dirichlet concentration: higher = more concentrated around true
            # Use concentration ~ 20 * true_alpha for moderate prior
            concentration = 20.0 * true_alpha + 1.0  # +1 to avoid zeros

            alpha[node_name] = pm.Dirichlet(
                f"alpha_{node_name}",
                a=concentration,
            )

        # ── Deterministic flows ──
        # Compute all flows from throughputs and allocations
        flow_list = []
        for node_name in NODE_NAMES:
            n_outputs = len(NODE_DEFS[node_name]["outputs"])
            for i in range(n_outputs):
                f_i = T[node_name] * alpha[node_name][i]
                flow_list.append(f_i)

        flows = pt.stack(flow_list)
        pm.Deterministic("flows", flows)

        # ── Throughput consistency ──
        # For each process node: T_n = sum of inflows
        # inflow to node n = sum over all source nodes of T_src * alpha_src[idx]
        for node_name in PROCESS_NODES:
            inflows = INFLOW_MAP[node_name]
            inflow_sum = sum(
                T[src] * alpha[src][idx] for src, idx in inflows
            )
            pm.Normal(
                f"throughput_consistency_{node_name}",
                mu=T[node_name] - inflow_sum,
                sigma=0.1,  # tight: throughput must match inflows
                observed=0.0,
            )

        # ── Flow observations ──
        if approach == "RSS":
            # One likelihood per flow
            pm.Normal(
                "y_obs",
                mu=flows,
                sigma=pt.as_tensor_variable(SIGMA_RSS),
                observed=OBSERVED_FLOWS,
            )
        else:
            # Independent: D likelihood terms per flow
            obs_y_long = []
            obs_sigma_long = []
            obs_flow_idx = []
            for j in range(N_FLOWS):
                for dim in DIMENSIONS:
                    obs_y_long.append(OBSERVED_FLOWS[j])
                    obs_sigma_long.append(SIGMAS_INDEP[j][dim])
                    obs_flow_idx.append(j)

            obs_y_long = np.array(obs_y_long)
            obs_sigma_long = np.array(obs_sigma_long)
            obs_flow_idx = np.array(obs_flow_idx, dtype=int)

            pm.Normal(
                "y_obs_dim",
                mu=flows[obs_flow_idx],
                sigma=pt.as_tensor_variable(obs_sigma_long),
                observed=obs_y_long,
            )

        # ── Ratio observation ──
        # BF allocation: alpha_BF[0]/(alpha_BF[0]+alpha_BF[1]) ≈ 0.9
        # This is a direct observation on the allocation coefficients!
        r = RATIO_OBS
        node_name = r["node"]
        outputs = NODE_DEFS[node_name]["outputs"]
        num_idx = outputs.index(r["numerator_output"])
        den_indices = [outputs.index(d) for d in r["denominator_outputs"]]

        ratio_computed = alpha[node_name][num_idx] / sum(
            alpha[node_name][k] for k in den_indices
        )
        pm.Normal(
            "ratio_obs",
            mu=ratio_computed,
            sigma=r["sigma_ratio"],
            observed=r["observed_ratio"],
        )

        # ── Sample ──
        idata = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
            cores=min(4, N_CHAINS), target_accept=0.95,
            random_seed=SEED, progressbar=True,
            init="adapt_diag",
        )

    print(f"\n[{tag}] Sampling complete.")
    return idata


# ═══════════════════════════════════════════════════════════════════════════
# 6.  RUN MODELS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("\n" + "=" * 90)
    print("RUNNING RSS MODEL (node-allocation parameterization)")
    print("=" * 90)
    idata_rss = build_and_sample_node_alloc("RSS", "RSS")

    print("\n" + "=" * 90)
    print("RUNNING INDEPENDENT MODEL (node-allocation parameterization)")
    print("=" * 90)
    idata_indep = build_and_sample_node_alloc("INDEP", "INDEP")

    # ═══════════════════════════════════════════════════════════════════════
    # 7.  EXTRACT AND COMPARE
    # ═══════════════════════════════════════════════════════════════════════

    def extract_flow_stats(idata, label):
        post = idata.posterior["flows"]
        flat = post.values.reshape(-1, N_FLOWS)
        records = []
        for j in range(N_FLOWS):
            samples = flat[:, j]
            lo, hi = az.hdi(samples, hdi_prob=0.95)
            records.append({
                "flow": FLOW_NAMES[j],
                "true": TRUE_FLOW_ARRAY[j],
                "observed": OBSERVED_FLOWS[j],
                f"mean_{label}": np.mean(samples),
                f"std_{label}": np.std(samples),
                f"hdi_lo_{label}": lo,
                f"hdi_hi_{label}": hi,
                f"hdi_width_{label}": hi - lo,
                f"error_{label}": np.mean(samples) - TRUE_FLOW_ARRAY[j],
                f"abs_error_{label}": abs(np.mean(samples) - TRUE_FLOW_ARRAY[j]),
            })
        return pd.DataFrame(records)

    def extract_throughput_stats(idata, label):
        records = []
        for node_name in NODE_NAMES:
            post = idata.posterior[f"T_{node_name}"]
            samples = post.values.flatten()
            lo, hi = az.hdi(samples, hdi_prob=0.95)
            records.append({
                "node": node_name,
                "true_T": TRUE_THROUGHPUTS[node_name],
                f"mean_T_{label}": np.mean(samples),
                f"std_T_{label}": np.std(samples),
                f"hdi_width_T_{label}": hi - lo,
            })
        return pd.DataFrame(records)

    def extract_allocation_stats(idata, label):
        records = []
        for node_name in NODE_NAMES:
            post = idata.posterior[f"alpha_{node_name}"]
            flat = post.values.reshape(-1, len(NODE_DEFS[node_name]["outputs"]))
            for i, dest in enumerate(NODE_DEFS[node_name]["outputs"]):
                samples = flat[:, i]
                lo, hi = az.hdi(samples, hdi_prob=0.95)
                records.append({
                    "node": node_name,
                    "output": dest,
                    "true_alpha": TRUE_ALLOCATIONS[node_name][i],
                    f"mean_alpha_{label}": np.mean(samples),
                    f"std_alpha_{label}": np.std(samples),
                    f"hdi_width_alpha_{label}": hi - lo,
                })
        return pd.DataFrame(records)

    # Extract stats
    df_flows_rss = extract_flow_stats(idata_rss, "rss")
    df_flows_indep = extract_flow_stats(idata_indep, "indep")
    df_flows = df_flows_rss.merge(
        df_flows_indep.drop(columns=["flow", "true", "observed"]),
        left_index=True, right_index=True,
    )

    df_T_rss = extract_throughput_stats(idata_rss, "rss")
    df_T_indep = extract_throughput_stats(idata_indep, "indep")
    df_T = df_T_rss.merge(
        df_T_indep.drop(columns=["node", "true_T"]),
        left_index=True, right_index=True,
    )

    df_alpha_rss = extract_allocation_stats(idata_rss, "rss")
    df_alpha_indep = extract_allocation_stats(idata_indep, "indep")
    df_alpha = df_alpha_rss.merge(
        df_alpha_indep.drop(columns=["node", "output", "true_alpha"]),
        left_index=True, right_index=True,
    )

    # Print results
    print("\n" + "=" * 90)
    print("POSTERIOR FLOW COMPARISON")
    print("=" * 90)
    print(df_flows.to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 90)
    print("POSTERIOR THROUGHPUT COMPARISON")
    print("=" * 90)
    print(df_T.to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 90)
    print("POSTERIOR ALLOCATION COMPARISON")
    print("=" * 90)
    print(df_alpha.to_string(index=False, float_format="%.4f"))

    # Check mass balance at posterior mean
    print("\n" + "=" * 90)
    print("MASS BALANCE CHECK AT POSTERIOR MEAN")
    print("=" * 90)
    for label, idata in [("RSS", idata_rss), ("INDEP", idata_indep)]:
        print(f"\n  [{label}]:")
        mean_flows = {}
        for j, fn in enumerate(FLOW_NAMES):
            post = idata.posterior["flows"]
            flat = post.values.reshape(-1, N_FLOWS)
            mean_flows[fn] = np.mean(flat[:, j])
        check_mass_balance(mean_flows)

    # ═══════════════════════════════════════════════════════════════════════
    # 8.  SAVE AND PLOT
    # ═══════════════════════════════════════════════════════════════════════

    OUT_DIR = "case_study_node_alloc_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    df_flows.to_csv(os.path.join(OUT_DIR, "flow_comparison.csv"), index=False)
    df_T.to_csv(os.path.join(OUT_DIR, "throughput_comparison.csv"), index=False)
    df_alpha.to_csv(os.path.join(OUT_DIR, "allocation_comparison.csv"), index=False)

    # Plot: posterior flows vs true
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(N_FLOWS)
    w = 0.2
    ax.bar(x_pos - 1.5*w, TRUE_FLOW_ARRAY, w, label="True", color="green", alpha=0.7)
    ax.bar(x_pos - 0.5*w, OBSERVED_FLOWS, w, label="Observed", color="gray", alpha=0.5)
    ax.bar(x_pos + 0.5*w, df_flows["mean_rss"], w,
           label="RSS", color="steelblue", alpha=0.8,
           yerr=df_flows["std_rss"], capsize=2)
    ax.bar(x_pos + 1.5*w, df_flows["mean_indep"], w,
           label="Indep", color="coral", alpha=0.8,
           yerr=df_flows["std_indep"], capsize=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FLOW_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Flow (Mt)")
    ax.set_title("Node-Allocation Model: Posterior Flows vs True")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "flows_comparison.png"), dpi=150)
    plt.close(fig)

    # Plot: allocations
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, node_name in enumerate(NODE_NAMES):
        ax = axes[idx // 3, idx % 3]
        mask = df_alpha["node"] == node_name
        sub = df_alpha[mask]
        x_pos_a = np.arange(len(sub))
        w_a = 0.25
        ax.bar(x_pos_a - w_a, sub["true_alpha"], w_a,
               label="True", color="green", alpha=0.7)
        ax.bar(x_pos_a, sub["mean_alpha_rss"], w_a,
               label="RSS", color="steelblue", alpha=0.8,
               yerr=sub["std_alpha_rss"], capsize=3)
        ax.bar(x_pos_a + w_a, sub["mean_alpha_indep"], w_a,
               label="Indep", color="coral", alpha=0.8,
               yerr=sub["std_alpha_indep"], capsize=3)
        ax.set_xticks(x_pos_a)
        ax.set_xticklabels(sub["output"].values, fontsize=8)
        ax.set_title(f"{node_name} allocations", fontsize=10)
        ax.set_ylim(0, 1.05)
        if idx == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Posterior Allocation Coefficients", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "allocations_comparison.png"), dpi=150)
    plt.close(fig)

    # Diagnostics
    print("\n" + "=" * 90)
    print("MCMC DIAGNOSTICS")
    print("=" * 90)
    for label, idata in [("RSS", idata_rss), ("INDEP", idata_indep)]:
        var_names = (
            [f"T_{n}" for n in NODE_NAMES] +
            [f"alpha_{n}" for n in NODE_NAMES]
        )
        summ = az.summary(idata, var_names=var_names, hdi_prob=0.95)
        summ.to_csv(os.path.join(OUT_DIR, f"summary_{label.lower()}.csv"))
        rhat_max = summ["r_hat"].max()
        ess_min = summ["ess_bulk"].min()
        print(f"  [{label}]  R-hat max: {rhat_max:.4f}   "
              f"ESS_bulk min: {ess_min:.0f}")

    print(f"\nAll outputs saved to: {OUT_DIR}/")

    print("\n" + "=" * 90)
    print("KEY ADVANTAGE OF NODE-ALLOCATION PARAMETERIZATION")
    print("=" * 90)
    print("""
    1. Mass balance is AUTOMATIC — allocations sum to 1 by Dirichlet prior.
    2. Ratio observations become DIRECT constraints on allocation coefficients.
    3. No need for soft mass-balance likelihood terms.
    4. Throughput consistency links nodes together.
    5. The ratio "90% to BOF, 10% to EAF" is simply an observation on
       alpha_BF[0] / (alpha_BF[0] + alpha_BF[1]).
    """)