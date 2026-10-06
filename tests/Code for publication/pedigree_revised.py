#!/usr/bin/env python3
"""
pedigree_revised.py  -  Dimension-wise pedigree uncertainty

Each pedigree dimension (coverage, frequency, spatial_boundary) produces
its own uncertainty distribution.  The per-dimension sigma_d values are
combined via root-sum-of-squares (RSS) under an independence assumption,
but are also stored individually so that scenario analysis can perturb
a single dimension without affecting the others.

Rubric dimensions (each scored 1-4, lower = better):
  coverage, frequency, spatial_boundary
"""

import numpy as np
import pandas as pd
from io_utils import load_csv, save_csv, ensure_dir

# -- per-dimension defaults --------------------------------------------------
DIM_PARAMS = {
    "coverage": {
        "r_min": 0.05,   "r_max": 0.40,
        "kappa_min": 5.0, "kappa_max": 200.0,
    },
    "frequency": {
        "r_min": 0.05,   "r_max": 0.35,
        "kappa_min": 5.0, "kappa_max": 200.0,
    },
    "spatial_boundary": {
        "r_min": 0.05,   "r_max": 0.30,
        "kappa_min": 5.0, "kappa_max": 200.0,
    },
}

EPSILON = 1e-6
SCORE_BEST = 1
SCORE_WORST = 4
DIMENSIONS = list(DIM_PARAMS.keys())


# -- per-dimension quality index ----------------------------------------------
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
            out[d] = 0.0
    return out


# -- per-dimension uncertainty ------------------------------------------------
def dim_relative_half_range(q_d: float, dim: str) -> float:
    r"""r_d(q_d) = r_max^(d) - q_d * (r_max^(d) - r_min^(d))."""
    p = DIM_PARAMS[dim]
    return p["r_max"] - q_d * (p["r_max"] - p["r_min"])


def dim_sigma(q_d: float, dim: str, y: float) -> float:
    r"""sigma_{i,d} = r_d(q_d) * max(|y|, epsilon)."""
    r = dim_relative_half_range(q_d, dim)
    return r * max(abs(y), EPSILON)


def dim_dirichlet_kappa(q_d: float, dim: str) -> float:
    r"""kappa_d = kappa_min^(d) + q_d * (kappa_max^(d) - kappa_min^(d))."""
    p = DIM_PARAMS[dim]
    return p["kappa_min"] + q_d * (p["kappa_max"] - p["kappa_min"])


# -- combined uncertainty (RSS) -----------------------------------------------
def combined_sigma(q_dict: dict, y: float) -> float:
    r"""
    sigma_i = |y| * sqrt(sum_d  r_d(q_d)^2)

    Independent error sources combine in quadrature (GUM / Heijungs 2014).
    """
    ss = sum(dim_relative_half_range(q, d) ** 2
             for d, q in q_dict.items())
    return max(abs(y), EPSILON) * np.sqrt(ss)


def combined_kappa(q_dict: dict) -> float:
    r"""
    Harmonic mean of per-dimension kappa_d values.
    The weakest dimension dominates, which is conservative.

    kappa = (1/D * sum_d  kappa_d^{-1})^{-1}
    """
    kappas = [dim_dirichlet_kappa(q, d) for d, q in q_dict.items()]
    if not kappas:
        return DIM_PARAMS[DIMENSIONS[0]]["kappa_min"]
    inv_mean = np.mean([1.0 / k for k in kappas])
    return 1.0 / inv_mean


# -- batch processing ---------------------------------------------------------
def score_dataframe(df: pd.DataFrame,
                    value_col: str = "y_median") -> pd.DataFrame:
    r"""
    Add per-dimension AND combined uncertainty columns to *df*.

    New columns added
    -----------------
    Per dimension d in {coverage, frequency, spatial_boundary}:
        q_{d}           quality index for dimension d
        sigma_{d}       observation sigma from dimension d alone
        kappa_{d}       Dirichlet kappa from dimension d alone

    Combined:
        sigma_combined      RSS-combined sigma
        kappa_combined      harmonically combined kappa
    """
    df = df.copy()
    y_vals = (pd.to_numeric(df.get(value_col, np.nan), errors="coerce")
              .fillna(0.0).to_numpy())

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
        df[f"kappa_{d}"] = df[q_col].apply(
            lambda q, dim=d: dim_dirichlet_kappa(q, dim)
        )

    # --- combined columns ---
    def _row_combined_sigma(idx):
        q_dict = {d: df.at[idx, f"q_{d}"] for d in DIMENSIONS}
        return combined_sigma(q_dict, y_vals[df.index.get_loc(idx)])

    def _row_combined_kappa(idx):
        q_dict = {d: df.at[idx, f"q_{d}"] for d in DIMENSIONS}
        return combined_kappa(q_dict)

    df["sigma_combined"] = [_row_combined_sigma(i) for i in df.index]
    df["kappa_combined"] = [_row_combined_kappa(i) for i in df.index]

    return df


# -- scenario helpers ---------------------------------------------------------
def override_dimension(df: pd.DataFrame,
                       dim: str,
                       new_score: int,
                       value_col: str = "y_median") -> pd.DataFrame:
    """Re-score after changing ONE pedigree dimension for all rows."""
    df = df.copy()
    df[dim] = new_score
    return score_dataframe(df, value_col=value_col)


def override_dimension_rows(df: pd.DataFrame,
                            dim: str,
                            new_score: int,
                            mask: pd.Series,
                            value_col: str = "y_median") -> pd.DataFrame:
    """Re-score a SUBSET of rows for one dimension."""
    df = df.copy()
    df.loc[mask, dim] = new_score
    return score_dataframe(df, value_col=value_col)


# -- CLI entry point ----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Dimension-wise pedigree scoring"
    )
    ap.add_argument("--input",
                    default="input data 2017/flow_data_score.csv")
    ap.add_argument("--output",
                    default="input data 2017/flow_data_score_mapped.csv")
    args = ap.parse_args()

    df = load_csv(args.input,
                  numeric_cols=DIMENSIONS + ["obs_std", "Value1"])
    if "y_median" not in df.columns:
        df["y_median"] = pd.to_numeric(
            df.get("Value1", np.nan), errors="coerce"
        )

    df = score_dataframe(df, value_col="y_median")
    save_csv(df, args.output)

    # -- summary --
    print("Pedigree scoring complete (dimension-wise).")
    print(f"  Dimensions:  {DIMENSIONS}")
    print(f"  Rows scored: {len(df)}")
    for d in DIMENSIONS:
        print(f"  sigma_{d}  range: "
              f"[{df[f'sigma_{d}'].min():.4f}, {df[f'sigma_{d}'].max():.4f}]")
    print(f"  sigma_combined   range: "
          f"[{df['sigma_combined'].min():.4f}, "
          f"{df['sigma_combined'].max():.4f}]")
    print(f"  kappa_combined   range: "
          f"[{df['kappa_combined'].min():.1f}, "
          f"{df['kappa_combined'].max():.1f}]")