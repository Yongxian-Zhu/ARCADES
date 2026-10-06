#!/usr/bin/env python3
"""
test_pedigree_revised.py
Test case for dimension-wise pedigree scoring using a subset of
U.S. iron and steelmaking flow data (USGS 2023 / Zhu et al. references).

Reads pedigree scores from  test_pedigree.xlsx  which has columns:
    Flow, Source, Coverage, Freq, Spatial, Rationale, Value (kt)

Usage
-----
    python test_pedigree_revised.py
    python test_pedigree_revised.py --input path/to/test_pedigree.xlsx
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the module under test.
# Stub io_utils if it is not on the path so pedigree_revised can be imported.
# ---------------------------------------------------------------------------
try:
    import io_utils                       # noqa: F401
except ModuleNotFoundError:
    import types
    io_utils = types.ModuleType("io_utils")
    io_utils.load_csv   = lambda *a, **kw: None
    io_utils.save_csv   = lambda *a, **kw: None
    io_utils.ensure_dir = lambda *a, **kw: None
    sys.modules["io_utils"] = io_utils

import pedigree_revised as ped


# ===================================================================
# 0.  PARSE ARGUMENTS
# ===================================================================
ap = argparse.ArgumentParser(
    description="Test dimension-wise pedigree scoring"
)
ap.add_argument(
    "--input",
    default=str(Path(__file__).parent / "test_pedigree.xlsx"),
    help="Path to the Excel file with pedigree scores",
)
ap.add_argument(
    "--output", default=None,
    help="Optional path to save scored results (.xlsx or .csv)",
)
args = ap.parse_args()

INPUT_PATH = Path(args.input)
if not INPUT_PATH.exists():
    sys.exit(f"ERROR: Input file not found: {INPUT_PATH.resolve()}")


# ===================================================================
# 1.  LOAD THE TEST DATA
# ===================================================================
print("=" * 72)
print(f"Loading test data from: {INPUT_PATH.resolve()}")
print("=" * 72)

df_raw = pd.read_excel(INPUT_PATH)

# -- Flexible column mapping ------------------------------------------
COLUMN_MAP = {
    # flow identification
    "flow":        "flow",
    "row_name":    "flow",
    "from":        "flow",
    # destination
    "to":          "col_name",
    "col_name":    "col_name",
    "destination": "col_name",
    # value
    "value (kt)":  "y_median",
    "value":       "y_median",
    "y_median":    "y_median",
    # pedigree dimensions
    "coverage":          "coverage",
    "freq":              "frequency",
    "frequency":         "frequency",
    "spatial":           "spatial_boundary",
    "spatial_boundary":  "spatial_boundary",
    # metadata
    "source":    "source",
    "rationale": "rationale",
}

rename = {}
for col in df_raw.columns:
    key = col.strip().lower()
    if key in COLUMN_MAP:
        rename[col] = COLUMN_MAP[key]
df_raw.rename(columns=rename, inplace=True)

# If the table has a single "Flow" column like "A -> B", split it.
if "col_name" not in df_raw.columns and "flow" in df_raw.columns:
    if df_raw["flow"].str.contains(r"[→\->]", regex=True).any():
        parts = df_raw["flow"].str.split(r"\s*[→\-\>]+\s*", n=1, expand=True)
        df_raw["row_name"] = parts[0].str.strip()
        df_raw["col_name"] = (parts[1].str.strip()
                              if parts.shape[1] > 1 else "")
    else:
        df_raw["row_name"] = df_raw["flow"]
        df_raw["col_name"] = ""

# Ensure the three pedigree columns are numeric
for dim in ped.DIMENSIONS:
    if dim not in df_raw.columns:
        sys.exit(
            f"ERROR: Required column '{dim}' not found in "
            f"{INPUT_PATH.name}.\n"
            f"  Available columns: {list(df_raw.columns)}"
        )
    df_raw[dim] = pd.to_numeric(df_raw[dim], errors="coerce")

# Ensure y_median exists and is numeric
if "y_median" not in df_raw.columns:
    numeric_candidates = [
        c for c in df_raw.columns
        if pd.api.types.is_numeric_dtype(df_raw[c])
        and c not in ped.DIMENSIONS
    ]
    if numeric_candidates:
        df_raw["y_median"] = df_raw[numeric_candidates[0]]
        print(f"  (Using '{numeric_candidates[0]}' as y_median)")
    else:
        sys.exit(
            "ERROR: No value column found. "
            "Please add a 'Value (kt)' or 'y_median' column."
        )

df_raw["y_median"] = (
    pd.to_numeric(df_raw["y_median"], errors="coerce").fillna(0.0)
)

print(f"  Loaded {len(df_raw)} flow records")
print(f"  Columns: {list(df_raw.columns)}")
print()


# ===================================================================
# 2.  SCORE THE DATA
# ===================================================================
df_scored = ped.score_dataframe(df_raw, value_col="y_median")


# ===================================================================
# 3.  UNIT-LEVEL ASSERTIONS
# ===================================================================
print("=" * 72)
print("UNIT TESTS")
print("=" * 72)

# 3a. dim_quality_index
assert ped.dim_quality_index(1) == 1.0,  "Score 1 -> q = 1.0"
assert ped.dim_quality_index(4) == 0.0,  "Score 4 -> q = 0.0"
assert abs(ped.dim_quality_index(2) - 2 / 3) < 1e-9
assert abs(ped.dim_quality_index(3) - 1 / 3) < 1e-9
print("  [PASS] dim_quality_index")

# 3b. dim_relative_half_range at boundary quality values
for d in ped.DIMENSIONS:
    r_best  = ped.dim_relative_half_range(1.0, d)
    r_worst = ped.dim_relative_half_range(0.0, d)
    assert abs(r_best  - ped.DIM_PARAMS[d]["r_min"]) < 1e-9, \
        f"{d}: r(q=1) should equal r_min"
    assert abs(r_worst - ped.DIM_PARAMS[d]["r_max"]) < 1e-9, \
        f"{d}: r(q=0) should equal r_max"
print("  [PASS] dim_relative_half_range boundary values")

# 3c. dim_sigma proportional to |y|
y_test = 1000.0
for d in ped.DIMENSIONS:
    s1 = ped.dim_sigma(1.0, d, y_test)
    s0 = ped.dim_sigma(0.0, d, y_test)
    assert s1 < s0, f"{d}: best quality should give smaller sigma"
    assert abs(s1 - ped.DIM_PARAMS[d]["r_min"] * y_test) < 1e-6
    assert abs(s0 - ped.DIM_PARAMS[d]["r_max"] * y_test) < 1e-6
print("  [PASS] dim_sigma")

# 3d. dim_dirichlet_kappa
for d in ped.DIMENSIONS:
    k_best  = ped.dim_dirichlet_kappa(1.0, d)
    k_worst = ped.dim_dirichlet_kappa(0.0, d)
    assert abs(k_best  - ped.DIM_PARAMS[d]["kappa_max"]) < 1e-9
    assert abs(k_worst - ped.DIM_PARAMS[d]["kappa_min"]) < 1e-9
    assert k_best > k_worst
print("  [PASS] dim_dirichlet_kappa")

# 3e. combined_sigma is RSS
q_all_best = {d: 1.0 for d in ped.DIMENSIONS}
expected_rss = y_test * np.sqrt(
    sum(ped.DIM_PARAMS[d]["r_min"] ** 2 for d in ped.DIMENSIONS)
)
assert abs(ped.combined_sigma(q_all_best, y_test) - expected_rss) < 1e-6
print("  [PASS] combined_sigma (RSS)")

# 3f. combined_kappa is harmonic mean
q_all_worst = {d: 0.0 for d in ped.DIMENSIONS}
kappas_worst = [ped.DIM_PARAMS[d]["kappa_min"] for d in ped.DIMENSIONS]
expected_hm = 1.0 / np.mean([1.0 / k for k in kappas_worst])
assert abs(ped.combined_kappa(q_all_worst) - expected_hm) < 1e-6
print("  [PASS] combined_kappa (harmonic mean)")

# 3g. All expected columns present in scored DataFrame
expected_cols = []
for d in ped.DIMENSIONS:
    expected_cols += [f"q_{d}", f"sigma_{d}", f"kappa_{d}"]
expected_cols += ["sigma_combined", "kappa_combined"]
for c in expected_cols:
    assert c in df_scored.columns, f"Missing column: {c}"
print("  [PASS] All output columns present")

# 3h. No NaN in computed columns
for c in expected_cols:
    assert df_scored[c].notna().all(), f"NaN found in {c}"
print("  [PASS] No NaN in output columns")

# 3i. sigma_combined >= max(sigma_d) for every row (RSS property)
for idx in df_scored.index:
    sig_dims = [df_scored.at[idx, f"sigma_{d}"] for d in ped.DIMENSIONS]
    assert df_scored.at[idx, "sigma_combined"] >= max(sig_dims) - 1e-9, \
        f"Row {idx}: sigma_combined should be >= max(sigma_d)"
print("  [PASS] sigma_combined >= max(sigma_d) for all rows")

print("\nAll unit tests passed.\n")


# ===================================================================
# 4.  PRINT SCORED TABLE
# ===================================================================
print("=" * 72)
print("SCORED FLOW DATA")
print("=" * 72)

id_cols = [
    c for c in ["row_name", "col_name", "flow", "y_median",
                "coverage", "frequency", "spatial_boundary"]
    if c in df_scored.columns
]
display_cols = id_cols + expected_cols

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.2f}".format)
print(df_scored[display_cols].to_string(index=False))


# ===================================================================
# 5.  SCENARIO 1
#     "What if literature-based flows had annual data (frequency -> 1)?"
# ===================================================================
print("\n" + "=" * 72)
print("SCENARIO 1: Improve frequency to 1 (annual) for literature-based flows")
print("=" * 72)

lit_mask = df_scored["frequency"] > 1
n_affected = lit_mask.sum()
print(f"  Flows affected: {n_affected}")

if n_affected > 0:
    df_scenario = ped.override_dimension_rows(
        df_scored, dim="frequency", new_score=1, mask=lit_mask,
        value_col="y_median",
    )

    flow_label = (
        "row_name" if "row_name" in df_scored.columns
        else "flow" if "flow" in df_scored.columns
        else None
    )

    comp_dict = {
        "y_median":         df_scored.loc[lit_mask, "y_median"].values,
        "sig_comb_before":  df_scored.loc[lit_mask, "sigma_combined"].values,
        "sig_comb_after":   df_scenario.loc[lit_mask, "sigma_combined"].values,
        "sig_freq_before":  df_scored.loc[lit_mask, "sigma_frequency"].values,
        "sig_freq_after":   df_scenario.loc[lit_mask, "sigma_frequency"].values,
        "kap_comb_before":  df_scored.loc[lit_mask, "kappa_combined"].values,
        "kap_comb_after":   df_scenario.loc[lit_mask, "kappa_combined"].values,
    }
    if flow_label:
        comp_dict = {
            flow_label: df_scored.loc[lit_mask, flow_label].values,
            **comp_dict,
        }

    comparison = pd.DataFrame(comp_dict)
    comparison["sig_reduction_%"] = (
        100 * (1 - comparison["sig_comb_after"] / comparison["sig_comb_before"])
    )
    print(comparison.to_string(index=False))

    # Assertions
    assert (
        comparison["sig_comb_after"] <= comparison["sig_comb_before"] + 1e-9
    ).all()
    print("\n  [PASS] sigma_combined decreased (or unchanged) for all "
          "affected flows")

    unaffected = ~lit_mask
    if unaffected.any():
        assert np.allclose(
            df_scored.loc[unaffected, "sigma_combined"].values,
            df_scenario.loc[unaffected, "sigma_combined"].values,
        ), "Unaffected rows should not change"
        print("  [PASS] Unaffected flows unchanged")
else:
    print("  (No flows with frequency > 1 found; skipping scenario)")


# ===================================================================
# 6.  SCENARIO 2
#     "What if the worst spatial_boundary flows improve to 1?"
# ===================================================================
print("\n" + "=" * 72)
print("SCENARIO 2: Improve worst spatial_boundary scores to 1")
print("=" * 72)

worst_spatial = (
    df_scored["spatial_boundary"] == df_scored["spatial_boundary"].max()
)
n_affected2 = worst_spatial.sum()
print(f"  Flows affected: {n_affected2}")

if n_affected2 > 0:
    df_scenario2 = ped.override_dimension_rows(
        df_scored, dim="spatial_boundary", new_score=1, mask=worst_spatial,
        value_col="y_median",
    )

    for idx in df_scored.index[worst_spatial]:
        label = ""
        if "row_name" in df_scored.columns:
            label += str(df_scored.at[idx, "row_name"])
        if "col_name" in df_scored.columns:
            label += " -> " + str(df_scored.at[idx, "col_name"])

        print(f"\n  {label}")
        print(
            f"    Before:  sig_spatial="
            f"{df_scored.at[idx, 'sigma_spatial_boundary']:.2f}"
            f"   sig_combined="
            f"{df_scored.at[idx, 'sigma_combined']:.2f}"
            f"   kap_combined="
            f"{df_scored.at[idx, 'kappa_combined']:.2f}"
        )
        print(
            f"    After:   sig_spatial="
            f"{df_scenario2.at[idx, 'sigma_spatial_boundary']:.2f}"
            f"   sig_combined="
            f"{df_scenario2.at[idx, 'sigma_combined']:.2f}"
            f"   kap_combined="
            f"{df_scenario2.at[idx, 'kappa_combined']:.2f}"
        )

        # sigma_coverage and sigma_frequency should be unchanged
        assert abs(
            df_scored.at[idx, "sigma_coverage"]
            - df_scenario2.at[idx, "sigma_coverage"]
        ) < 1e-9, f"Row {idx}: sigma_coverage should not change"
        assert abs(
            df_scored.at[idx, "sigma_frequency"]
            - df_scenario2.at[idx, "sigma_frequency"]
        ) < 1e-9, f"Row {idx}: sigma_frequency should not change"

    print("\n  [PASS] sigma_coverage and sigma_frequency unchanged")
    print("  [PASS] Only spatial_boundary dimension affected")
else:
    print("  (No flows to adjust; skipping scenario)")


# ===================================================================
# 7.  SCENARIO 3 (GLOBAL)
#     "What if ALL coverage scores improve by one level?"
# ===================================================================
print("\n" + "=" * 72)
print("SCENARIO 3: Improve all coverage scores by 1 level (clamped at 1)")
print("=" * 72)

df_scenario3 = df_scored.copy()
df_scenario3["coverage"] = (df_scenario3["coverage"] - 1).clip(lower=1)
df_scenario3 = ped.score_dataframe(df_scenario3, value_col="y_median")

print(f"  Mean sig_coverage  before: "
      f"{df_scored['sigma_coverage'].mean():.2f}")
print(f"  Mean sig_coverage  after:  "
      f"{df_scenario3['sigma_coverage'].mean():.2f}")
print(f"  Mean sig_combined  before: "
      f"{df_scored['sigma_combined'].mean():.2f}")
print(f"  Mean sig_combined  after:  "
      f"{df_scenario3['sigma_combined'].mean():.2f}")

assert (df_scenario3["sigma_coverage"].mean()
        <= df_scored["sigma_coverage"].mean() + 1e-9)
print("  [PASS] Mean sigma_coverage decreased or unchanged")

assert np.allclose(
    df_scored["sigma_frequency"].values,
    df_scenario3["sigma_frequency"].values,
)
assert np.allclose(
    df_scored["sigma_spatial_boundary"].values,
    df_scenario3["sigma_spatial_boundary"].values,
)
print("  [PASS] sigma_frequency and sigma_spatial_boundary unchanged")


# ===================================================================
# 8.  SUMMARY STATISTICS
# ===================================================================
print("\n" + "=" * 72)
print("SUMMARY STATISTICS")
print("=" * 72)
for d in ped.DIMENSIONS:
    col = f"sigma_{d}"
    print(f"  sigma_{d:20s}  min={df_scored[col].min():10.2f}"
          f"  max={df_scored[col].max():10.2f}"
          f"  mean={df_scored[col].mean():10.2f}")
print(f"  {'sigma_combined':24s}  min={df_scored['sigma_combined'].min():10.2f}"
      f"  max={df_scored['sigma_combined'].max():10.2f}"
      f"  mean={df_scored['sigma_combined'].mean():10.2f}")
print(f"  {'kappa_combined':24s}  min={df_scored['kappa_combined'].min():10.2f}"
      f"  max={df_scored['kappa_combined'].max():10.2f}"
      f"  mean={df_scored['kappa_combined'].mean():10.2f}")


# ===================================================================
# 9.  OPTIONAL: SAVE RESULTS
# ===================================================================
if args.output:
    out_path = Path(args.output)
    if out_path.suffix in (".xlsx", ".xls"):
        df_scored.to_excel(out_path, index=False)
    else:
        df_scored.to_csv(out_path, index=False)
    print(f"\n  Scored results saved to: {out_path.resolve()}")


print("\n" + "=" * 72)
print("ALL TESTS AND SCENARIOS COMPLETED SUCCESSFULLY")
print("=" * 72)

# ===================================================================
# 10.  SCORE THE DATA
# ===================================================================
df_scored = ped.score_dataframe(df_raw, value_col="y_median")

# Save scored results next to the input file
output_path = INPUT_PATH.parent / INPUT_PATH.stem
df_scored.to_excel(str(output_path) + "_scored.xlsx", index=False)
df_scored.to_csv(str(output_path) + "_scored.csv", index=False)
print(f"  Scored results saved to:")
print(f"    {output_path}_scored.xlsx")
print(f"    {output_path}_scored.csv")
print()