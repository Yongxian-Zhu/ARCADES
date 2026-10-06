#!/usr/bin/env python3
"""
pedigree_independent.py - Each pedigree dimension as an independent measurement

Instead of combining per-dimension uncertainties via RSS, each pedigree
dimension (coverage, frequency, spatial_boundary) is treated as an
independent pseudo-measurement of the same underlying true value.

In a Bayesian model the likelihood for data point i becomes:

    L(theta_i | y_i) = prod_d  N(y_i | theta_i, sigma_{i,d}^2)

which is equivalent to a single Gaussian with precision equal to the
SUM of per-dimension precisions:

    tau_eff = sum_d  1 / sigma_{i,d}^2
    sigma_eff = 1 / sqrt(tau_eff)

This is *not* the same as RSS of sigmas.  It treats each quality
dimension as an independent corroborating observation rather than an
independent error source.

For Dirichlet concentration (compositional data), the analogous
operation is additive pooling of kappa across dimensions.

Rubric dimensions (each scored 1-4, lower = better):
    coverage, frequency, spatial_boundary
"""

import numpy as np
import pandas as pd
from io_utils import load_csv, save_csv, ensure_dir

# ---------------------------------------------------------------------------
# Per-dimension parameter ranges (identical to the RSS version so that
# the per-dimension sigma_d and kappa_d values are directly comparable)
# ---------------------------------------------------------------------------
DIM_PARAMS = {
    "coverage": {
        "r_min": 0.05,
        "r_max": 0.40,
        "kappa_min": 5.0,
        "kappa_max": 200.0,
    },
    "frequency": {
        "r_min": 0.05,
        "r_max": 0.35,
        "kappa_min": 5.0,
        "kappa_max": 200.0,
    },
    "spatial_boundary": {
        "r_min": 0.05,
        "r_max": 0.30,
        "kappa_min": 5.0,
        "kappa_max": 200.0,
    },
}

EPSILON = 1e-6
SCORE_BEST = 1
SCORE_WORST = 4
DIMENSIONS = list(DIM_PARAMS.keys())


# ---------------------------------------------------------------------------
# Per-dimension quality index  (unchanged from RSS version)
# ---------------------------------------------------------------------------
def dim_quality_index(score: float) -> float:
    """Map a single ordinal score (1=best, 4=worst) to q_d in [0, 1]."""
    q = (SCORE_WORST - score) / (SCORE_WORST - SCORE_BEST)
    return float(np.clip(q, 0.0, 1.0))


def quality_indices(row: pd.Series) -> dict:
    """Return {dim: q_d} for every scored dimension."""
    out = {}
    for d in DIMENSIONS:
        v = row.get(d, np.nan)
        if pd.notna(v):
            out[d] = dim_quality_index(float(v))
        else:
            out[d] = 0.0          # missing score → worst quality
    return out


# ---------------------------------------------------------------------------
# Per-dimension uncertainty  (unchanged from RSS version)
# ---------------------------------------------------------------------------
def dim_relative_half_range(q_d: float, dim: str) -> float:
    r"""r_d(q_d) = r_max^{(d)} - q_d · (r_max^{(d)} - r_min^{(d)})."""
    p = DIM_PARAMS[dim]
    return p["r_max"] - q_d * (p["r_max"] - p["r_min"])


def dim_sigma(q_d: float, dim: str, y: float) -> float:
    r"""sigma_{i,d} = r_d(q_d) · \max(|y|, \epsilon)."""
    r = dim_relative_half_range(q_d, dim)
    return r * max(abs(y), EPSILON)


def dim_dirichlet_kappa(q_d: float, dim: str) -> float:
    r"""kappa_d = kappa_min^{(d)} + q_d · (kappa_max^{(d)} - kappa_min^{(d)})."""
    p = DIM_PARAMS[dim]
    return p["kappa_min"] + q_d * (p["kappa_max"] - p["kappa_min"])


# ---------------------------------------------------------------------------
# Per-dimension precision (tau = 1/sigma^2)
# ---------------------------------------------------------------------------
def dim_precision(q_d: float, dim: str, y: float) -> float:
    """tau_{i,d} = 1 / sigma_{i,d}^2."""
    s = dim_sigma(q_d, dim, y)
    return 1.0 / (s ** 2)


# ---------------------------------------------------------------------------
# Combined uncertainty — independent-measurement (precision-sum) approach
# ---------------------------------------------------------------------------
def combined_precision(q_dict: dict, y: float) -> float:
    r"""
    tau_eff = ∑_d  1 / sigma_{i,d}^2

    Each dimension contributes an independent likelihood factor, so
    precisions add.
    """
    return sum(dim_precision(q, d, y) for d, q in q_dict.items())


def combined_sigma_from_precision(q_dict: dict, y: float) -> float:
    r"""
    sigma_eff = 1 / sqrt(tau_eff)

    This is the effective observation noise that a single-Gaussian
    likelihood would need to reproduce the same posterior update as
    the D independent likelihoods.
    """
    tau = combined_precision(q_dict, y)
    if tau <= 0:
        # Fallback: return the largest per-dimension sigma
        return max(
            dim_sigma(q, d, y) for d, q in q_dict.items()
        ) if q_dict else max(abs(y), EPSILON)
    return 1.0 / np.sqrt(tau)


def combined_kappa_additive(q_dict: dict) -> float:
    r"""
    kappa_eff = ∑_d  kappa_d

    For Dirichlet likelihoods the analogous "independent measurement"
    combination is additive: each dimension's kappa adds to the total
    concentration, reflecting increased confidence from multiple
    corroborating quality assessments.

    Contrast with the RSS version which uses the *harmonic mean*
    (conservative, dominated by the weakest dimension).
    """
    kappas = [dim_dirichlet_kappa(q, d) for d, q in q_dict.items()]
    if not kappas:
        return DIM_PARAMS[DIMENSIONS[0]]["kappa_min"]
    return sum(kappas)


# ---------------------------------------------------------------------------
# Measurement-list builder (for models that literally want D observations)
# ---------------------------------------------------------------------------
def expand_to_measurements(
    df: pd.DataFrame,
    value_col: str = "y_median",
    id_col: str | None = None,
) -> pd.DataFrame:
    r"""
    Expand each row into D rows, one per pedigree dimension.

    Returns a long-format DataFrame with columns:
        - all original columns
        - ``pedigree_dim``   : which dimension this row represents
        - ``sigma_obs``      : sigma_{i,d} for this dimension
        - ``kappa_obs``      : kappa_d for this dimension
        - ``tau_obs``        : precision 1/sigma_{i,d}^2
        - ``measurement_id`` : unique id tying replicates to the same
                               original data point

    This is the format a Stan / PyMC / NumPyro model can consume
    directly: each row is one "observation" with its own noise level.
    """
    y_vals = (
        pd.to_numeric(df.get(value_col, np.nan), errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    records = []
    for iloc_idx, (df_idx, row) in enumerate(df.iterrows()):
        y = y_vals[iloc_idx]
        q_dict = quality_indices(row)
        base = row.to_dict()

        for d in DIMENSIONS:
            q_d = q_dict.get(d, 0.0)
            s_d = dim_sigma(q_d, d, y)
            k_d = dim_dirichlet_kappa(q_d, d)

            rec = dict(base)
            rec["pedigree_dim"] = d
            rec["sigma_obs"] = s_d
            rec["kappa_obs"] = k_d
            rec["tau_obs"] = 1.0 / (s_d ** 2)
            rec["measurement_id"] = (
                row[id_col] if id_col and id_col in row.index
                else df_idx
            )
            records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Batch processing — wide format (one row per data point, like RSS version)
# ---------------------------------------------------------------------------
def score_dataframe(
    df: pd.DataFrame,
    value_col: str = "y_median",
) -> pd.DataFrame:
    r"""
    Add per-dimension AND combined uncertainty columns to *df*.

    New columns added
    -----------------
    Per dimension d in {coverage, frequency, spatial_boundary}:
        q_{d}       quality index for dimension d
        sigma_{d}   observation sigma from dimension d alone
        kappa_{d}   Dirichlet kappa from dimension d alone
        tau_{d}     precision from dimension d alone

    Combined (independent-measurement / precision-sum):
        tau_combined     sum of per-dimension precisions
        sigma_combined   1 / sqrt(tau_combined)
        kappa_combined   sum of per-dimension kappas

    Comparison helpers:
        sigma_rss        what the RSS approach would have given
    """
    df = df.copy()
    y_vals = (
        pd.to_numeric(df.get(value_col, np.nan), errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    # --- per-dimension columns ---
    for d in DIMENSIONS:
        q_col = f"q_{d}"
        df[q_col] = df.apply(
            lambda row, dim=d: dim_quality_index(
                float(row[dim]) if pd.notna(row.get(dim)) else SCORE_WORST
            ),
            axis=1,
        )
        df[f"sigma_{d}"] = [
            dim_sigma(q, d, y)
            for q, y in zip(df[q_col], y_vals)
        ]
        df[f"tau_{d}"] = 1.0 / (df[f"sigma_{d}"] ** 2)
        df[f"kappa_{d}"] = df[q_col].apply(
            lambda q, dim=d: dim_dirichlet_kappa(q, dim)
        )

    # --- combined columns (precision-sum) ---
    def _row_q_dict(idx):
        return {d: df.at[idx, f"q_{d}"] for d in DIMENSIONS}

    tau_combined = []
    sigma_combined = []
    kappa_combined = []
    sigma_rss = []

    for pos, idx in enumerate(df.index):
        q_dict = _row_q_dict(idx)
        y = y_vals[pos]

        # Precision-sum approach
        tau_eff = combined_precision(q_dict, y)
        tau_combined.append(tau_eff)
        sigma_combined.append(1.0 / np.sqrt(tau_eff) if tau_eff > 0 else np.inf)
        kappa_combined.append(combined_kappa_additive(q_dict))

        # RSS for comparison
        ss = sum(dim_relative_half_range(q, d) ** 2 for d, q in q_dict.items())
        sigma_rss.append(max(abs(y), EPSILON) * np.sqrt(ss))

    df["tau_combined"] = tau_combined
    df["sigma_combined"] = sigma_combined
    df["kappa_combined"] = kappa_combined
    df["sigma_rss"] = sigma_rss

    return df


# ---------------------------------------------------------------------------
# Scenario helpers  (same interface as RSS version)
# ---------------------------------------------------------------------------
def override_dimension(
    df: pd.DataFrame,
    dim: str,
    new_score: int,
    value_col: str = "y_median",
) -> pd.DataFrame:
    """Re-score after changing ONE pedigree dimension for all rows."""
    df = df.copy()
    df[dim] = new_score
    return score_dataframe(df, value_col=value_col)


def override_dimension_rows(
    df: pd.DataFrame,
    dim: str,
    new_score: int,
    mask: pd.Series,
    value_col: str = "y_median",
) -> pd.DataFrame:
    """Re-score a SUBSET of rows for one dimension."""
    df = df.copy()
    df.loc[mask, dim] = new_score
    return score_dataframe(df, value_col=value_col)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Pedigree scoring — independent-measurement approach"
    )
    ap.add_argument("--input",
                    default="input data 2017/flow_data_score.csv")
    ap.add_argument("--output",
                    default="input data 2017/flow_data_score_mapped_indep.csv")
    ap.add_argument("--expand", action="store_true",
                    help="Also write a long-format file with D rows per data point")
    ap.add_argument("--expand-output",
                    default="input data 2017/flow_data_score_expanded.csv")
    args = ap.parse_args()

    df = load_csv(args.input, numeric_cols=DIMENSIONS + ["obs_std", "Value1"])
    if "y_median" not in df.columns:
        df["y_median"] = pd.to_numeric(
            df.get("Value1", np.nan), errors="coerce"
        )

    # --- wide-format scoring ---
    df_scored = score_dataframe(df, value_col="y_median")
    save_csv(df_scored, args.output)

    # --- optional long-format expansion ---
    if args.expand:
        df_long = expand_to_measurements(df_scored, value_col="y_median")
        save_csv(df_long, args.expand_output)
        print(f"  Expanded file : {args.expand_output}  "
              f"({len(df_long)} rows = {len(df)} × {len(DIMENSIONS)})")

    # --- summary ---
    print("Pedigree scoring complete (independent-measurement approach).")
    print(f"  Dimensions       : {DIMENSIONS}")
    print(f"  Rows scored      : {len(df_scored)}")
    for d in DIMENSIONS:
        lo = df_scored[f"sigma_{d}"].min()
        hi = df_scored[f"sigma_{d}"].max()
        print(f"  sigma_{d:20s} range: [{lo:.4f}, {hi:.4f}]")

    print(f"  sigma_combined (precision-sum) range: "
          f"[{df_scored['sigma_combined'].min():.4f}, "
          f"{df_scored['sigma_combined'].max():.4f}]")
    print(f"  sigma_rss      (for comparison)      range: "
          f"[{df_scored['sigma_rss'].min():.4f}, "
          f"{df_scored['sigma_rss'].max():.4f}]")
    print(f"  kappa_combined (additive)     range: "
          f"[{df_scored['kappa_combined'].min():.1f}, "
          f"{df_scored['kappa_combined'].max():.1f}]")
    print(f"  tau_combined                  range: "
          f"[{df_scored['tau_combined'].min():.2f}, "
          f"{df_scored['tau_combined'].max():.2f}]")