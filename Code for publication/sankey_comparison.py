#!/usr/bin/env python3
"""
sankey_comparison.py

Reads posterior summary results from the RSS and INDEP Bayesian reconciliation
models and draws side-by-side Sankey diagrams comparing the two methods.

Usage
-----
    python sankey_comparison.py
    python sankey_comparison.py --rss results_rss.csv --indep results_indep.csv
    python sankey_comparison.py --output sankey_comparison.png

Input format (CSV):
    Each CSV should have columns: flow_idx, from, to, true, observed, post_mean, post_sd
    
    If no CSV files are provided, the script will look for ArviZ InferenceData
    NetCDF files or use example/demo data.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
from pathlib import Path

# Try importing plotly for interactive Sankey (preferred)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Note: plotly not installed. Using matplotlib for Sankey diagrams.")
    print("  Install plotly for better interactive diagrams: pip install plotly")


# ===================================================================
# 0.  PARSE ARGUMENTS
# ===================================================================
ap = argparse.ArgumentParser(
    description="Draw Sankey diagrams comparing RSS and INDEP reconciliation"
)
ap.add_argument(
    "--rss", default=None,
    help="Path to RSS model results CSV",
)
ap.add_argument(
    "--indep", default=None,
    help="Path to INDEP model results CSV",
)
ap.add_argument(
    "--output", default="sankey_comparison",
    help="Output filename prefix (without extension)",
)
ap.add_argument(
    "--format", default="both", choices=["html", "png", "pdf", "both"],
    help="Output format: html (interactive), png, pdf, or both",
)
args = ap.parse_args()


# ===================================================================
# 1.  DEFINE THE NETWORK STRUCTURE
# ===================================================================
# Flow definitions: (index, source_node, target_node)
FLOW_DEFS = [
    (0,  "Iron Ore",  "BF"),
    (1,  "Iron Ore",  "DR"),
    (2,  "BF",        "BOF"),
    (3,  "DR",        "EAF"),
    (4,  "Scrap",     "BOF"),
    (5,  "Scrap",     "EAF"),
    (6,  "BOF",       "Steel Out"),
    (7,  "EAF",       "Steel Out"),
    (8,  "BF",        "BF Losses"),
    (9,  "DR",        "DR Losses"),
    (10, "BOF",       "BOF Losses"),
    (11, "EAF",       "EAF Losses"),
]

# All unique node names (in display order)
NODE_NAMES = [
    "Iron Ore", "Scrap",           # Sources
    "BF", "DR",                     # Primary processing
    "BOF", "EAF",                   # Secondary processing
    "Steel Out",                    # Final product
    "BF Losses", "DR Losses",      # Losses
    "BOF Losses", "EAF Losses",
]

# Node colors
NODE_COLORS = {
    "Iron Ore":    "#4e79a7",
    "Scrap":       "#59a14f",
    "BF":          "#f28e2b",
    "DR":          "#e15759",
    "BOF":         "#76b7b2",
    "EAF":         "#edc948",
    "Steel Out":   "#b07aa1",
    "BF Losses":   "#ff9da7",
    "DR Losses":   "#9c755f",
    "BOF Losses":  "#bab0ac",
    "EAF Losses":  "#d4a6c8",
}

# True flows (ground truth for reference)
TRUE_FLOWS = np.array([
    130.0, 30.0, 85.0, 32.0, 15.0, 40.0,
    96.0, 70.0, 45.0, 2.0, 4.0, 2.0,
])


# ===================================================================
# 2.  LOAD OR GENERATE DATA
# ===================================================================
def load_results_csv(filepath):
    """Load results from a CSV file."""
    df = pd.read_csv(filepath)
    # Standardize column names
    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if "post_mean" in key or "posterior_mean" in key:
            col_map[col] = "post_mean"
        elif "post_sd" in key or "posterior_sd" in key:
            col_map[col] = "post_sd"
        elif key in ("true", "true_flow", "true_value"):
            col_map[col] = "true"
        elif key in ("observed", "obs", "y_obs"):
            col_map[col] = "observed"
        elif key in ("from", "source", "from_node"):
            col_map[col] = "from"
        elif key in ("to", "target", "to_node"):
            col_map[col] = "to"
        elif key in ("flow_idx", "idx", "index"):
            col_map[col] = "flow_idx"
        elif key in ("hdi_3%", "hdi_low", "ci_low"):
            col_map[col] = "hdi_low"
        elif key in ("hdi_97%", "hdi_high", "ci_high"):
            col_map[col] = "hdi_high"
    df.rename(columns=col_map, inplace=True)
    return df


def generate_demo_data():
    """
    Generate demo reconciliation results for both methods.
    These simulate what the Bayesian models would produce.
    """
    np.random.seed(42)

    # Observed values (noisy version of true flows)
    noise_scale = np.array([
        0.05, 0.08, 0.07, 0.10, 0.06, 0.05,
        0.04, 0.09, 0.12, 0.15, 0.08, 0.10,
    ])
    observed = TRUE_FLOWS * (1 + noise_scale * np.random.randn(12))

    # --- RSS model results (wider posteriors, larger adjustments) ---
    # RSS has larger sigma, so posterior means are pulled more toward
    # mass-balance-consistent values but with wider uncertainty
    rss_sigma = np.array([
        11.24, 6.82, 17.40, 12.59, 3.23, 6.51,
        8.52, 22.68, 14.90, 1.20, 1.58, 2.12,
    ])

    # Simulate posterior means (adjusted toward mass balance)
    # The adjustment is proportional to the constraint violation
    # weighted by the inverse variance
    rss_post_mean = observed.copy()
    # Apply mass-balance corrections (simplified)
    # BF: x0 - x2 - x8 = 0
    residual_bf = rss_post_mean[0] - rss_post_mean[2] - rss_post_mean[8]
    weights_bf = 1.0 / rss_sigma[[0, 2, 8]]**2
    weights_bf /= weights_bf.sum()
    rss_post_mean[0] -= residual_bf * weights_bf[0]
    rss_post_mean[2] += residual_bf * weights_bf[1]
    rss_post_mean[8] += residual_bf * weights_bf[2]

    # DR: x1 - x3 - x9 = 0
    residual_dr = rss_post_mean[1] - rss_post_mean[3] - rss_post_mean[9]
    weights_dr = 1.0 / rss_sigma[[1, 3, 9]]**2
    weights_dr /= weights_dr.sum()
    rss_post_mean[1] -= residual_dr * weights_dr[0]
    rss_post_mean[3] += residual_dr * weights_dr[1]
    rss_post_mean[9] += residual_dr * weights_dr[2]

    # BOF: x2 + x4 - x6 - x10 = 0
    residual_bof = rss_post_mean[2] + rss_post_mean[4] - rss_post_mean[6] - rss_post_mean[10]
    weights_bof = 1.0 / rss_sigma[[2, 4, 6, 10]]**2
    weights_bof /= weights_bof.sum()
    rss_post_mean[2] -= residual_bof * weights_bof[0]
    rss_post_mean[4] -= residual_bof * weights_bof[1]
    rss_post_mean[6] += residual_bof * weights_bof[2]
    rss_post_mean[10] += residual_bof * weights_bof[3]

    # EAF: x3 + x5 - x7 - x11 = 0
    residual_eaf = rss_post_mean[3] + rss_post_mean[5] - rss_post_mean[7] - rss_post_mean[11]
    weights_eaf = 1.0 / rss_sigma[[3, 5, 7, 11]]**2
    weights_eaf /= weights_eaf.sum()
    rss_post_mean[3] -= residual_eaf * weights_eaf[0]
    rss_post_mean[5] -= residual_eaf * weights_eaf[1]
    rss_post_mean[7] += residual_eaf * weights_eaf[2]
    rss_post_mean[11] += residual_eaf * weights_eaf[3]

    rss_post_sd = rss_sigma * 0.7  # Posterior is narrower than prior

    # --- INDEP model results (narrower posteriors, smaller adjustments) ---
    indep_sigma_eff = np.array([
        3.75, 1.35, 3.76, 3.41, 0.66, 1.35,
        2.84, 6.63, 4.35, 0.35, 0.46, 0.70,
    ])

    indep_post_mean = observed.copy()
    # Similar mass-balance correction but with tighter sigmas
    residual_bf = indep_post_mean[0] - indep_post_mean[2] - indep_post_mean[8]
    weights_bf = 1.0 / indep_sigma_eff[[0, 2, 8]]**2
    weights_bf /= weights_bf.sum()
    indep_post_mean[0] -= residual_bf * weights_bf[0]
    indep_post_mean[2] += residual_bf * weights_bf[1]
    indep_post_mean[8] += residual_bf * weights_bf[2]

    residual_dr = indep_post_mean[1] - indep_post_mean[3] - indep_post_mean[9]
    weights_dr = 1.0 / indep_sigma_eff[[1, 3, 9]]**2
    weights_dr /= weights_dr.sum()
    indep_post_mean[1] -= residual_dr * weights_dr[0]
    indep_post_mean[3] += residual_dr * weights_dr[1]
    indep_post_mean[9] += residual_dr * weights_dr[2]

    residual_bof = indep_post_mean[2] + indep_post_mean[4] - indep_post_mean[6] - indep_post_mean[10]
    weights_bof = 1.0 / indep_sigma_eff[[2, 4, 6, 10]]**2
    weights_bof /= weights_bof.sum()
    indep_post_mean[2] -= residual_bof * weights_bof[0]
    indep_post_mean[4] -= residual_bof * weights_bof[1]
    indep_post_mean[6] += residual_bof * weights_bof[2]
    indep_post_mean[10] += residual_bof * weights_bof[3]

    residual_eaf = indep_post_mean[3] + indep_post_mean[5] - indep_post_mean[7] - indep_post_mean[11]
    weights_eaf = 1.0 / indep_sigma_eff[[3, 5, 7, 11]]**2
    weights_eaf /= weights_eaf.sum()
    indep_post_mean[3] -= residual_eaf * weights_eaf[0]
    indep_post_mean[5] -= residual_eaf * weights_eaf[1]
    indep_post_mean[7] += residual_eaf * weights_eaf[2]
    indep_post_mean[11] += residual_eaf * weights_eaf[3]

    indep_post_sd = indep_sigma_eff * 0.7

    # Build DataFrames
    rows_rss = []
    rows_indep = []
    for i, (idx, src, tgt) in enumerate(FLOW_DEFS):
        rows_rss.append({
            "flow_idx": idx, "from": src, "to": tgt,
            "true": TRUE_FLOWS[i], "observed": observed[i],
            "post_mean": rss_post_mean[i], "post_sd": rss_post_sd[i],
            "hdi_low": rss_post_mean[i] - 1.96 * rss_post_sd[i],
            "hdi_high": rss_post_mean[i] + 1.96 * rss_post_sd[i],
        })
        rows_indep.append({
            "flow_idx": idx, "from": src, "to": tgt,
            "true": TRUE_FLOWS[i], "observed": observed[i],
            "post_mean": indep_post_mean[i], "post_sd": indep_post_sd[i],
            "hdi_low": indep_post_mean[i] - 1.96 * indep_post_sd[i],
            "hdi_high": indep_post_mean[i] + 1.96 * indep_post_sd[i],
        })

    return pd.DataFrame(rows_rss), pd.DataFrame(rows_indep)


# Load data
if args.rss and args.indep:
    print(f"Loading RSS results from: {args.rss}")
    print(f"Loading INDEP results from: {args.indep}")
    df_rss = load_results_csv(args.rss)
    df_indep = load_results_csv(args.indep)
else:
    print("No input files specified. Generating demo data...")
    print("  (Use --rss and --indep to provide actual results)")
    df_rss, df_indep = generate_demo_data()

# Print summary tables
print("\n" + "=" * 90)
print("RSS MODEL RESULTS")
print("=" * 90)
print(df_rss.to_string(index=False, float_format="{:.2f}".format))

print("\n" + "=" * 90)
print("INDEP MODEL RESULTS")
print("=" * 90)
print(df_indep.to_string(index=False, float_format="{:.2f}".format))


# ===================================================================
# 3.  COMPARISON TABLE
# ===================================================================
print("\n" + "=" * 90)
print("COMPARISON: RSS vs INDEP")
print("=" * 90)

comp = pd.DataFrame({
    "flow": [f"{r['from']} → {r['to']}" for _, r in df_rss.iterrows()],
    "true": df_rss["true"].values,
    "observed": df_rss["observed"].values,
    "rss_mean": df_rss["post_mean"].values,
    "rss_sd": df_rss["post_sd"].values,
    "indep_mean": df_indep["post_mean"].values,
    "indep_sd": df_indep["post_sd"].values,
})
comp["rss_err"] = comp["rss_mean"] - comp["true"]
comp["indep_err"] = comp["indep_mean"] - comp["true"]
comp["rss_abs_err"] = comp["rss_err"].abs()
comp["indep_abs_err"] = comp["indep_err"].abs()
comp["sd_ratio"] = comp["rss_sd"] / comp["indep_sd"]

print(comp.to_string(index=False, float_format="{:.2f}".format))

print(f"\nMean absolute error  RSS:   {comp['rss_abs_err'].mean():.2f}")
print(f"Mean absolute error  INDEP: {comp['indep_abs_err'].mean():.2f}")
print(f"Mean SD ratio (RSS/INDEP):  {comp['sd_ratio'].mean():.2f}")


# ===================================================================
# 4.  PLOTLY SANKEY DIAGRAMS (Interactive)
# ===================================================================
def make_plotly_sankey(df, title, node_names, node_colors_dict, flow_defs):
    """Create a Plotly Sankey diagram from reconciliation results."""
    node_idx = {name: i for i, name in enumerate(node_names)}

    sources = []
    targets = []
    values = []
    labels_flow = []
    colors_flow = []

    for _, row in df.iterrows():
        src = row["from"]
        tgt = row["to"]
        val = max(row["post_mean"], 0.1)  # Sankey needs positive values

        sources.append(node_idx[src])
        targets.append(node_idx[tgt])
        values.append(val)

        # Color based on adjustment direction
        obs_val = row.get("observed", val)
        if val > obs_val * 1.02:
            colors_flow.append("rgba(44, 160, 44, 0.5)")   # Green: increased
        elif val < obs_val * 0.98:
            colors_flow.append("rgba(214, 39, 40, 0.5)")   # Red: decreased
        else:
            colors_flow.append("rgba(150, 150, 150, 0.5)")  # Gray: ~unchanged

        sd_str = f" ± {row['post_sd']:.1f}" if "post_sd" in row else ""
        labels_flow.append(f"{src} → {tgt}: {val:.1f}{sd_str} kt")

    node_colors_list = [node_colors_dict.get(n, "#888888") for n in node_names]

    fig = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=1),
            label=node_names,
            color=node_colors_list,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors_flow,
            customdata=labels_flow,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )
    return fig


if HAS_PLOTLY:
    print("\n" + "=" * 90)
    print("GENERATING PLOTLY SANKEY DIAGRAMS")
    print("=" * 90)

    # --- Side-by-side comparison ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "RSS Model (Single Combined σ)",
            "INDEP Model (Dimension-wise Observations)",
        ),
        specs=[[{"type": "sankey"}, {"type": "sankey"}]],
        horizontal_spacing=0.05,
    )

    sankey_rss = make_plotly_sankey(
        df_rss, "RSS", NODE_NAMES, NODE_COLORS, FLOW_DEFS
    )
    sankey_indep = make_plotly_sankey(
        df_indep, "INDEP", NODE_NAMES, NODE_COLORS, FLOW_DEFS
    )

    fig.add_trace(sankey_rss, row=1, col=1)
    fig.add_trace(sankey_indep, row=1, col=2)

    fig.update_layout(
        title_text=(
            "Bayesian Mass-Balance Reconciliation: RSS vs Independent Dimensions<br>"
            "<sub>Green links = increased from observed | "
            "Red links = decreased from observed | "
            "Gray links = ~unchanged</sub>"
        ),
        font_size=12,
        height=700,
        width=1400,
    )

    if args.format in ("html", "both"):
        html_path = f"{args.output}_comparison.html"
        fig.write_html(html_path)
        print(f"  Saved interactive Sankey: {html_path}")

    if args.format in ("png", "pdf", "both"):
        try:
            img_path = f"{args.output}_comparison.png"
            fig.write_image(img_path, scale=2)
            print(f"  Saved static Sankey: {img_path}")
        except Exception as e:
            print(f"  Could not save static image: {e}")
            print("  (Install kaleido: pip install kaleido)")

    # --- Individual detailed Sankey diagrams ---
    for label, df_model in [("RSS", df_rss), ("INDEP", df_indep)]:
        fig_single = go.Figure(data=[
            make_plotly_sankey(
                df_model, label, NODE_NAMES, NODE_COLORS, FLOW_DEFS
            )
        ])

        # Add annotations with flow values
        annotations_text = []
        for _, row in df_model.iterrows():
            true_val = row.get("true", None)
            true_str = f" (true: {true_val:.1f})" if true_val else ""
            annotations_text.append(
                f"{row['from']} → {row['to']}: "
                f"{row['post_mean']:.1f} ± {row['post_sd']:.1f}{true_str}"
            )

        fig_single.update_layout(
            title_text=(
                f"{label} Model — Reconciled Steelmaking Flows (kt)<br>"
                f"<sub>{'  |  '.join(annotations_text[:6])}</sub>"
            ),
            font_size=12,
            height=600,
            width=900,
        )

        if args.format in ("html", "both"):
            html_path = f"{args.output}_{label.lower()}.html"
            fig_single.write_html(html_path)
            print(f"  Saved {label} Sankey: {html_path}")


# ===================================================================
# 5.  MATPLOTLIB SANKEY DIAGRAMS (Static fallback)
# ===================================================================
def draw_matplotlib_sankey(ax, df, title, node_names, flow_defs):
    """
    Draw a simplified Sankey-style flow diagram using matplotlib.
    Since matplotlib's Sankey is limited, we use a custom approach
    with horizontal bars and arrows.
    """
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Define node positions (x, y)
    node_pos = {
        "Iron Ore":    (0.0, 0.75),
        "Scrap":       (0.0, 0.25),
        "BF":          (0.3, 0.85),
        "DR":          (0.3, 0.55),
        "BOF":         (0.6, 0.75),
        "EAF":         (0.6, 0.35),
        "Steel Out":   (0.9, 0.55),
        "BF Losses":   (0.5, 1.0),
        "DR Losses":   (0.5, 0.40),
        "BOF Losses":  (0.8, 0.95),
        "EAF Losses":  (0.8, 0.15),
    }

    # Scale factor for line widths
    max_flow = df["post_mean"].max()
    width_scale = 15.0 / max_flow

    # Draw flows as arrows
    for _, row in df.iterrows():
        src = row["from"]
        tgt = row["to"]
        val = row["post_mean"]
        obs = row.get("observed", val)

        x1, y1 = node_pos[src]
        x2, y2 = node_pos[tgt]

        # Color based on adjustment
        if val > obs * 1.02:
            color = "#2ca02c"  # Green: increased
            alpha = 0.6
        elif val < obs * 0.98:
            color = "#d62728"  # Red: decreased
            alpha = 0.6
        else:
            color = "#7f7f7f"  # Gray: unchanged
            alpha = 0.4

        lw = max(val * width_scale, 0.5)

        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=alpha,
                lw=lw,
                connectionstyle="arc3,rad=0.1",
                mutation_scale=15,
            ),
        )

        # Label on the arrow
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(
            mx, my, f"{val:.1f}",
            fontsize=7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )

    # Draw nodes
    for name, (x, y) in node_pos.items():
        color = NODE_COLORS.get(name, "#888888")
        ax.plot(x, y, "o", markersize=18, color=color, zorder=5,
                markeredgecolor="black", markeredgewidth=1)
        ax.text(x, y - 0.06, name, fontsize=7, ha="center", va="top",
                fontweight="bold")

    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(0.0, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


# Always generate matplotlib version as well
print("\n" + "=" * 90)
print("GENERATING MATPLOTLIB DIAGRAMS")
print("=" * 90)

# --- Figure 1: Side-by-side Sankey ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig1.suptitle(
    "Bayesian Mass-Balance Reconciliation: RSS vs Independent Dimensions\n"
    "Green = increased from observed | Red = decreased | Gray = ~unchanged",
    fontsize=14, fontweight="bold",
)

draw_matplotlib_sankey(ax1, df_rss, "RSS Model\n(Single Combined σ)",
                       NODE_NAMES, FLOW_DEFS)
draw_matplotlib_sankey(ax2, df_indep, "INDEP Model\n(Dimension-wise Observations)",
                       NODE_NAMES, FLOW_DEFS)

plt.tight_layout()
fig1.savefig(f"{args.output}_sankey_comparison.png", dpi=200, bbox_inches="tight")
fig1.savefig(f"{args.output}_sankey_comparison.pdf", bbox_inches="tight")
print(f"  Saved: {args.output}_sankey_comparison.png")
print(f"  Saved: {args.output}_sankey_comparison.pdf")


# --- Figure 2: Bar chart comparison ---
fig2, axes = plt.subplots(2, 1, figsize=(14, 10))

flow_labels = [f"{r['from']}→{r['to']}" for _, r in df_rss.iterrows()]
x = np.arange(len(flow_labels))
width = 0.2

# Top panel: Flow values
ax = axes[0]
ax.bar(x - 1.5 * width, TRUE_FLOWS, width, label="True", color="#4e79a7", alpha=0.8)
ax.bar(x - 0.5 * width, df_rss["observed"].values, width, label="Observed",
       color="#f28e2b", alpha=0.8)
ax.bar(x + 0.5 * width, df_rss["post_mean"].values, width, label="RSS Posterior",
       color="#e15759", alpha=0.8,
       yerr=df_rss["post_sd"].values, capsize=3)
ax.bar(x + 1.5 * width, df_indep["post_mean"].values, width, label="INDEP Posterior",
       color="#59a14f", alpha=0.8,
       yerr=df_indep["post_sd"].values, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(flow_labels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Flow (kt)")
ax.set_title("Flow Values: True vs Observed vs Reconciled", fontweight="bold")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Bottom panel: Errors
ax2 = axes[1]
ax2.bar(x - width, df_rss["observed"].values - TRUE_FLOWS, width,
        label="Observed Error", color="#f28e2b", alpha=0.7)
ax2.bar(x, df_rss["post_mean"].values - TRUE_FLOWS, width,
        label="RSS Error", color="#e15759", alpha=0.7)
ax2.bar(x + width, df_indep["post_mean"].values - TRUE_FLOWS, width,
        label="INDEP Error", color="#59a14f", alpha=0.7)

ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
ax2.set_xticks(x)
ax2.set_xticklabels(flow_labels, rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("Error (kt)")
ax2.set_title("Errors: Posterior Mean − True Value", fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig2.savefig(f"{args.output}_bar_comparison.png", dpi=200, bbox_inches="tight")
fig2.savefig(f"{args.output}_bar_comparison.pdf", bbox_inches="tight")
print(f"  Saved: {args.output}_bar_comparison.png")
print(f"  Saved: {args.output}_bar_comparison.pdf")


# --- Figure 3: Uncertainty comparison ---
fig3, ax3 = plt.subplots(figsize=(12, 6))

x = np.arange(len(flow_labels))
ax3.barh(x - 0.2, df_rss["post_sd"].values, 0.35,
         label="RSS Posterior SD", color="#e15759", alpha=0.7)
ax3.barh(x + 0.2, df_indep["post_sd"].values, 0.35,
         label="INDEP Posterior SD", color="#59a14f", alpha=0.7)

ax3.set_yticks(x)
ax3.set_yticklabels(flow_labels, fontsize=9)
ax3.set_xlabel("Posterior Standard Deviation (kt)")
ax3.set_title(
    "Uncertainty Comparison: RSS vs INDEP\n"
    f"(Mean ratio σ_RSS / σ_INDEP = {comp['sd_ratio'].mean():.1f}×)",
    fontweight="bold",
)
ax3.legend(loc="lower right")
ax3.grid(axis="x", alpha=0.3)
ax3.invert_yaxis()

plt.tight_layout()
fig3.savefig(f"{args.output}_uncertainty_comparison.png", dpi=200, bbox_inches="tight")
fig3.savefig(f"{args.output}_uncertainty_comparison.pdf", bbox_inches="tight")
print(f"  Saved: {args.output}_uncertainty_comparison.png")
print(f"  Saved: {args.output}_uncertainty_comparison.pdf")


# --- Figure 4: Credible intervals ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

for ax, df_model, label, color in [
    (ax4a, df_rss, "RSS", "#e15759"),
    (ax4b, df_indep, "INDEP", "#59a14f"),
]:
    y_pos = np.arange(len(flow_labels))

    # HDI bars
    if "hdi_low" in df_model.columns and "hdi_high" in df_model.columns:
        hdi_low = df_model["hdi_low"].values
        hdi_high = df_model["hdi_high"].values
    else:
        hdi_low = df_model["post_mean"].values - 1.96 * df_model["post_sd"].values
        hdi_high = df_model["post_mean"].values + 1.96 * df_model["post_sd"].values

    # Draw HDI bars
    for i in range(len(flow_labels)):
        ax.plot(
            [hdi_low[i], hdi_high[i]], [y_pos[i], y_pos[i]],
            color=color, linewidth=3, alpha=0.5, solid_capstyle="round",
        )

    # Posterior means
    ax.scatter(
        df_model["post_mean"].values, y_pos,
        color=color, s=60, zorder=5, label=f"{label} posterior mean",
        edgecolors="black", linewidths=0.5,
    )

    # True values
    ax.scatter(
        TRUE_FLOWS, y_pos,
        color="black", marker="D", s=40, zorder=6, label="True value",
    )

    # Observed values
    ax.scatter(
        df_model["observed"].values, y_pos,
        color="#f28e2b", marker="x", s=50, zorder=4, label="Observed",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(flow_labels, fontsize=9)
    ax.set_xlabel("Flow (kt)")
    ax.set_title(f"{label} Model — 95% Credible Intervals", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

fig4.suptitle(
    "Credible Intervals: RSS vs Independent Dimensions",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
fig4.savefig(f"{args.output}_credible_intervals.png", dpi=200, bbox_inches="tight")
fig4.savefig(f"{args.output}_credible_intervals.pdf", bbox_inches="tight")
print(f"  Saved: {args.output}_credible_intervals.png")
print(f"  Saved: {args.output}_credible_intervals.pdf")


# ===================================================================
# 6.  SUMMARY
# ===================================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"  Files generated:")
print(f"    {args.output}_sankey_comparison.png/pdf  — Side-by-side Sankey diagrams")
print(f"    {args.output}_bar_comparison.png/pdf     — Bar chart comparison")
print(f"    {args.output}_uncertainty_comparison.png/pdf — Uncertainty comparison")
print(f"    {args.output}_credible_intervals.png/pdf — Credible interval forest plots")
if HAS_PLOTLY:
    print(f"    {args.output}_comparison.html           — Interactive Plotly Sankey")
    print(f"    {args.output}_rss.html                  — Interactive RSS Sankey")
    print(f"    {args.output}_indep.html                — Interactive INDEP Sankey")

print(f"\n  Key findings:")
print(f"    RSS mean absolute error:   {comp['rss_abs_err'].mean():.2f} kt")
print(f"    INDEP mean absolute error: {comp['indep_abs_err'].mean():.2f} kt")
print(f"    Mean σ ratio (RSS/INDEP):  {comp['sd_ratio'].mean():.1f}×")
print(f"    RSS posteriors are ~{comp['sd_ratio'].mean():.1f}× wider than INDEP posteriors")

# Check mass balance
for label, df_model in [("RSS", df_rss), ("INDEP", df_indep)]:
    pm = df_model["post_mean"].values
    mb = np.array([
        pm[0] - pm[2] - pm[8],
        pm[1] - pm[3] - pm[9],
        pm[2] + pm[4] - pm[6] - pm[10],
        pm[3] + pm[5] - pm[7] - pm[11],
    ])
    print(f"    {label} mass balance residuals: {mb}")
    print(f"    {label} max |residual|: {np.max(np.abs(mb)):.4f}")

plt.show()
print("\nDone.")
