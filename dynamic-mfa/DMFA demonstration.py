import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Rectangle, FancyArrowPatch

# =============================================================================
# 1. Input data: Steel production / inventory scenario
# =============================================================================

years = np.arange(2018, 2051)

steel_inflow = np.array([
    86.6, 87.8, 72.7, 85.8, 80.5, 83.33333333, 86.16666667,
    89.0, 89.4, 89.8, 90.2, 90.6, 91.0, 91.6, 92.2, 92.8,
    93.4, 94.0, 94.8, 95.6, 96.4, 97.2, 98.0, 98.6, 99.2,
    99.8, 100.4, 101.0, 101.6, 102.2, 102.8, 103.4, 104.0
])

# Units are assumed to be million metric tons per year.
unit = "Mt steel/year"

# =============================================================================
# 2. DMFA core functions
# =============================================================================

class LifetimeModel:
    def __init__(self, mu=30, sigma=10, max_age=120):
        """
        Aggregated steel product lifetime model.
        mu: mean lifetime, years
        sigma: standard deviation, years
        max_age: maximum modeled age
        """
        self.mu = mu
        self.sigma = sigma
        self.max_age = max_age

    def survival(self, ages):
        ages = np.asarray(ages)
        cdf = norm.cdf(ages, loc=self.mu, scale=self.sigma)
        sr = 1 - cdf
        sr[ages > self.max_age] = 0
        return sr

    def retirement_pdf(self, ages):
        """
        Approximate discrete retirement probability by age.
        Probability that a cohort retires between age a-1 and age a.
        """
        ages = np.asarray(ages)
        cdf_now = norm.cdf(ages, loc=self.mu, scale=self.sigma)
        cdf_prev = norm.cdf(ages - 1, loc=self.mu, scale=self.sigma)
        pdf = np.maximum(cdf_now - cdf_prev, 0)
        pdf[ages > self.max_age] = 0
        return pdf


def flow_driven_dmfa(inflow, lifetime):
    """
    Cohort-based flow-driven DMFA.

    stock[t] = sum_i inflow[i] * survival[t-i]
    outflow[t] = inflow[t] - change_in_stock[t]
    """
    inflow = np.asarray(inflow)
    T = len(inflow)

    stock = np.zeros(T)
    outflow = np.zeros(T)

    ages = np.arange(T)
    sr = lifetime.survival(ages)

    for t in range(T):
        stock[t] = np.sum(inflow[:t+1] * sr[t::-1])

    outflow[0] = inflow[0] - stock[0]
    outflow[1:] = inflow[1:] - np.diff(stock)

    return stock, outflow


# =============================================================================
# 3. Warm-start / historical backcast
# =============================================================================

def build_backcast_series(years_future, inflow_future, start_year=1967,
                          method="flat_2018"):
    """
    Creates a historical inflow series before 2018 so that the model includes
    older cohorts generating scrap during 2018-2050.

    For final analysis, replace this with historical apparent steel consumption
    or product-resolved steel input data.
    """
    first_future_year = years_future[0]
    hist_years = np.arange(start_year, first_future_year)

    if method == "flat_2018":
        hist_inflow = np.full(len(hist_years), inflow_future[0])

    elif method == "linear_to_2018":
        # Example generic backcast: linearly increase from 70 Mt in 1967
        # to the 2018 value. Replace with actual historical data if available.
        hist_inflow = np.linspace(70, inflow_future[0], len(hist_years))

    else:
        raise ValueError("Unknown backcast method.")

    all_years = np.concatenate([hist_years, years_future])
    all_inflow = np.concatenate([hist_inflow, inflow_future])

    return all_years, all_inflow


# =============================================================================
# 4. Baseline steel DMFA run
# =============================================================================

# Recommended baseline for aggregated steel:
# Mean lifetime = 30 years, sigma = 10 years.
# This is an illustrative aggregate assumption. Product-resolved model should
# use separate lifetimes for construction, vehicles, machinery, etc.
baseline_lifetime = LifetimeModel(mu=30, sigma=10, max_age=120)

all_years, all_inflow = build_backcast_series(
    years,
    steel_inflow,
    start_year=1967,
    method="linear_to_2018"
)

stock_all, scrap_all = flow_driven_dmfa(all_inflow, baseline_lifetime)

# Extract presentation period
mask = all_years >= 2018
years_plot = all_years[mask]
inflow_plot = all_inflow[mask]
stock_plot = stock_all[mask]
scrap_plot = scrap_all[mask]

scrap_ratio = scrap_plot / inflow_plot
cumulative_scrap = np.cumsum(scrap_plot)

# =============================================================================
# 5. Monte Carlo uncertainty run
# =============================================================================

def monte_carlo_scrap(n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)

    scrap_samples = []
    stock_samples = []

    for i in range(n_iter):
        # Sample annual inflow uncertainty
        # 5% CV shown here. Adjust as needed.
        inflow_noise = rng.normal(1.0, 0.05, size=len(all_inflow))
        sampled_inflow = all_inflow * inflow_noise

        # Sample lifetime uncertainty
        # Aggregate steel lifetime uncertainty. Adjust with literature values.
        mu = rng.normal(30, 4)
        sigma = max(rng.normal(10, 2), 2)

        lt = LifetimeModel(mu=mu, sigma=sigma, max_age=120)

        stock_i, scrap_i = flow_driven_dmfa(sampled_inflow, lt)

        stock_samples.append(stock_i[mask])
        scrap_samples.append(scrap_i[mask])

    return np.array(stock_samples), np.array(scrap_samples)


stock_samples, scrap_samples = monte_carlo_scrap(n_iter=1000, seed=7)

scrap_p05 = np.percentile(scrap_samples, 5, axis=0)
scrap_p50 = np.percentile(scrap_samples, 50, axis=0)
scrap_p95 = np.percentile(scrap_samples, 95, axis=0)

ratio_samples = scrap_samples / inflow_plot
ratio_p05 = np.percentile(ratio_samples, 5, axis=0)
ratio_p50 = np.percentile(ratio_samples, 50, axis=0)
ratio_p95 = np.percentile(ratio_samples, 95, axis=0)

cum_scrap_samples = np.cumsum(scrap_samples, axis=1)
cum_p05 = np.percentile(cum_scrap_samples, 5, axis=0)
cum_p50 = np.percentile(cum_scrap_samples, 50, axis=0)
cum_p95 = np.percentile(cum_scrap_samples, 95, axis=0)

# =============================================================================
# 6. Plot styling
# =============================================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150
})

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
gray = "#666666"
light_orange = "#ffd8b1"


# =============================================================================
# Figure 1: Steel inflow / production projection
# =============================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.plot(years_plot, inflow_plot, color=blue, linewidth=3, marker="o", markersize=4)

ax.set_title("Steel Production / Inventory Scenario Used as DMFA Inflow")
ax.set_ylabel(unit)
ax.set_xlabel("Year")
ax.grid(True, alpha=0.3)
ax.set_xlim(2018, 2050)

ax.annotate(
    f"{inflow_plot[0]:.1f} Mt",
    xy=(2018, inflow_plot[0]),
    xytext=(2019, inflow_plot[0] + 5),
    arrowprops=dict(arrowstyle="->", color=gray),
    color=gray
)

ax.annotate(
    f"{inflow_plot[-1]:.1f} Mt",
    xy=(2050, inflow_plot[-1]),
    xytext=(2043, inflow_plot[-1] - 8),
    arrowprops=dict(arrowstyle="->", color=gray),
    color=gray
)

plt.tight_layout()
plt.savefig("fig_steel_inflow.png", bbox_inches="tight")
plt.savefig("fig_steel_inflow.svg", bbox_inches="tight")


# =============================================================================
# Figure 2: Baseline scrap generation vs production
# =============================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.plot(years_plot, inflow_plot, color=blue, linewidth=3, label="Steel production / inflow")
ax.plot(years_plot, scrap_plot, color=orange, linewidth=3, label="Modeled obsolete scrap generation")

ax.set_title("Flow-Driven DMFA: Steel Inflow and Modeled Obsolete Scrap")
ax.set_ylabel("Mt steel/year")
ax.set_xlabel("Year")
ax.grid(True, alpha=0.3)
ax.set_xlim(2018, 2050)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("fig_steel_scrap_generation.png", bbox_inches="tight")
plt.savefig("fig_steel_scrap_generation.svg", bbox_inches="tight")


# =============================================================================
# Figure 3: Scrap generation uncertainty band
# =============================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.fill_between(
    years_plot,
    scrap_p05,
    scrap_p95,
    color=light_orange,
    alpha=0.8,
    label="5th–95th percentile"
)

ax.plot(years_plot, scrap_p50, color=orange, linewidth=3, label="Median modeled scrap")
ax.plot(years_plot, inflow_plot, color=blue, linewidth=2.5, linestyle="--", label="Steel production / inflow")

ax.set_title("Modeled Obsolete Steel Scrap Generation with Uncertainty")
ax.set_ylabel("Mt steel/year")
ax.set_xlabel("Year")
ax.grid(True, alpha=0.3)
ax.set_xlim(2018, 2050)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("fig_steel_scrap_uncertainty.png", bbox_inches="tight")
plt.savefig("fig_steel_scrap_uncertainty.svg", bbox_inches="tight")


# =============================================================================
# Figure 4: Scrap-to-production ratio
# =============================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.fill_between(
    years_plot,
    ratio_p05 * 100,
    ratio_p95 * 100,
    color=light_orange,
    alpha=0.8,
    label="5th–95th percentile"
)

ax.plot(years_plot, ratio_p50 * 100, color=orange, linewidth=3, label="Median")
ax.axhline(100, color=gray, linestyle="--", linewidth=1.5, label="100% of annual production")

ax.set_title("Theoretical Obsolete Scrap-to-Production Ratio")
ax.set_ylabel("Modeled scrap / production (%)")
ax.set_xlabel("Year")
ax.grid(True, alpha=0.3)
ax.set_xlim(2018, 2050)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("fig_steel_scrap_ratio.png", bbox_inches="tight")
plt.savefig("fig_steel_scrap_ratio.svg", bbox_inches="tight")


# =============================================================================
# Figure 5: Cumulative scrap generation
# =============================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.fill_between(
    years_plot,
    cum_p05,
    cum_p95,
    color=light_orange,
    alpha=0.8,
    label="5th–95th percentile"
)

ax.plot(years_plot, cum_p50, color=orange, linewidth=3, label="Median cumulative scrap")

ax.set_title("Cumulative Modeled Obsolete Steel Scrap Generation")
ax.set_ylabel("Cumulative Mt steel")
ax.set_xlabel("Year")
ax.grid(True, alpha=0.3)
ax.set_xlim(2018, 2050)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("fig_steel_cumulative_scrap.png", bbox_inches="tight")
plt.savefig("fig_steel_cumulative_scrap.svg", bbox_inches="tight")


# =============================================================================
# Figure 6: DMFA framework diagram
# =============================================================================

def add_box(ax, xy, width, height, text, facecolor, edgecolor="#333333"):
    rect = Rectangle(xy, width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True
    )

def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.8,
        color="#333333"
    )
    ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(10, 5.8))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

add_box(
    ax, (0.3, 4.2), 2.0, 1.1,
    "Scenario Inputs\nProduction, demand,\ntechnology pathways,\nuncertainty",
    "#d9eaf7"
)

add_box(
    ax, (2.9, 4.2), 2.0, 1.1,
    "Material Config.\nProducts, alloys,\nlifetimes, collection,\ncomposition",
    "#e2f0d9"
)

add_box(
    ax, (5.5, 4.2), 2.0, 1.1,
    "DMFA Engine\nCohort accounting,\nsurvival functions,\nMonte Carlo",
    "#fff2cc"
)

add_box(
    ax, (8.1, 4.2), 1.6, 1.1,
    "Outputs\nStock, scrap,\nprimary demand,\nratios",
    "#f4cccc"
)

add_box(
    ax, (1.2, 2.1), 2.5, 1.0,
    "Flow-Driven Mode\nAnnual inflow → stock + scrap",
    "#fce5cd"
)

add_box(
    ax, (6.2, 2.1), 2.5, 1.0,
    "Stock-Driven Mode\nTarget stock → required inflow + scrap",
    "#fce5cd"
)

add_box(
    ax, (3.7, 0.4), 2.6, 1.0,
    "Adaptable to New Materials\nSwap configuration files:\ncommodity, product, lifetime,\ncomposition, scenarios",
    "#eadcf8"
)

add_arrow(ax, (2.3, 4.75), (2.9, 4.75))
add_arrow(ax, (4.9, 4.75), (5.5, 4.75))
add_arrow(ax, (7.5, 4.75), (8.1, 4.75))

add_arrow(ax, (6.5, 4.2), (2.45, 3.1))
add_arrow(ax, (6.5, 4.2), (7.45, 3.1))

add_arrow(ax, (2.45, 2.1), (5.0, 1.4))
add_arrow(ax, (7.45, 2.1), (5.0, 1.4))

ax.set_title("Generic Dynamic Material Flow Analysis Framework Developed for Pathways Enhancement", fontsize=14)

plt.tight_layout()
plt.savefig("fig_dmfa_framework.png", bbox_inches="tight")
plt.savefig("fig_dmfa_framework.svg", bbox_inches="tight")


# =============================================================================
# 7. Export summary table for slide
# =============================================================================

summary_years = [2018, 2025, 2030, 2040, 2050]
summary = []

for y in summary_years:
    idx = np.where(years_plot == y)[0][0]
    summary.append({
        "Year": y,
        "Steel production / inflow, Mt": inflow_plot[idx],
        "Modeled scrap p50, Mt": scrap_p50[idx],
        "Modeled scrap p05, Mt": scrap_p05[idx],
        "Modeled scrap p95, Mt": scrap_p95[idx],
        "Scrap-to-production p50, %": ratio_p50[idx] * 100,
        "Cumulative scrap p50, Mt": cum_p50[idx]
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("steel_dmfa_summary_table.csv", index=False)

print(summary_df.round(2))
print("\nFigures saved as PNG and SVG files.")