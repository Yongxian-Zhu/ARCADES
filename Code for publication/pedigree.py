#!/usr/bin/env python3
"""
pedigree.py
Pedigree-based data-quality scoring and mapping to observation
uncertainty (σ for Gaussian, \kappa for Dirichlet).

Rubric dimensions (each scored 1–4, lower = better):
  coverage, frequency, spatial_boundary

Quality index  q ∈ [0, 1]  is computed as the normalised average of
the three dimension scores.  q is then mapped to:
  • relative half-range  r(q) for continuous observations, and
  • Dirichlet concentration  \kappa(q) for allocation observations.
"""

import numpy as np
import pandas as pd
from io_utils import load_csv, save_csv, ensure_dir

# ── defaults ────────────────────────────────────────────────────────
R_MIN = 0.05          # relative half-range at q = 1 (best data)
R_MAX = 1.00          # relative half-range at q = 0 (worst data)
KAPPA_MIN = 2.0       # Dirichlet concentration at q = 0
KAPPA_MAX = 200.0     # Dirichlet concentration at q = 1
EPSILON = 1e-6        # floor for very small observed values
SCORE_BEST = 1        # best ordinal score
SCORE_WORST = 4       # worst ordinal score
DIMENSIONS = ["coverage", "frequency", "spatial_boundary"]


# ── quality index ───────────────────────────────────────────────────
def quality_index(row: pd.Series, dims=DIMENSIONS) -> float:
    """Map ordinal scores (1=best … 4=worst) to q ∈ [0, 1]."""
    vals = []
    for d in dims:
        v = row.get(d, np.nan)
        if pd.notna(v):
            vals.append(float(v))
    if not vals:
        return 0.0                       # no info → lowest quality
    avg = np.mean(vals)                  # average ordinal score
    q = (SCORE_WORST - avg) / (SCORE_WORST - SCORE_BEST)
    return float(np.clip(q, 0.0, 1.0))


# ── mapping functions ───────────────────────────────────────────────
def relative_half_range(q: float) -> float:
    """r(q) = r_max − q·(r_max − r_min)."""
    return R_MAX - q * (R_MAX - R_MIN)


def obs_sigma(q: float, y: float) -> float:
    """σ_i = r(q) · max(|y|, \epsilon)."""
    r = relative_half_range(q)
    return r * max(abs(y), EPSILON)


def dirichlet_kappa(q: float) -> float:
    """\kappa = \kappa_min + q·(\kappa_max − \kappa_min)."""
    return KAPPA_MIN + q * (KAPPA_MAX - KAPPA_MIN)


# ── batch processing ────────────────────────────────────────────────
def score_dataframe(df: pd.DataFrame,
                    value_col: str = "y_median") -> pd.DataFrame:
    """Add columns  quality_index, obs_sigma, dirichlet_kappa  to *df*.

    *df* must contain pedigree dimension columns (coverage, frequency,
    spatial_boundary) and a representative observed value column.
    """
    df = df.copy()
    df["quality_index"] = df.apply(quality_index, axis=1)
    y_vals = pd.to_numeric(df.get(value_col, np.nan), errors="coerce"
                           ).fillna(0.0).to_numpy()
    df["obs_sigma"] = [
        obs_sigma(q, y) for q, y in zip(df["quality_index"], y_vals)
    ]
    df["dirichlet_kappa"] = df["quality_index"].apply(dirichlet_kappa)
    return df


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score pedigree table")
    ap.add_argument("--input", default="input data 2017/flow_data_score.csv")
    ap.add_argument("--output", default="input data 2017/flow_data_score_mapped.csv")
    args = ap.parse_args()

    df = load_csv(args.input,
                  numeric_cols=DIMENSIONS + ["obs_std", "Value1"])
    # use Value1 as representative y if obs_std not present
    if "y_median" not in df.columns:
        df["y_median"] = pd.to_numeric(df.get("Value1", np.nan),
                                       errors="coerce")
    df = score_dataframe(df, value_col="y_median")
    save_csv(df, args.output)
    print("Pedigree scoring complete.")