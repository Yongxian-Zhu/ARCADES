#!/usr/bin/env python3
"""
Generate ARCADE auxiliary input tables for a given year from flow_data_{year}.csv.

Inputs:
  input data {year}/flow_data_{year}.csv   (e.g., flow_data_2022.csv)

Outputs (created/overwritten):
  input data {year}/flow_data.csv                  (copy of flow_data_{year}.csv)
  input data {year}/nodes.csv
  input data {year}/flow_data_score.csv
  input data {year}/bounds.csv
  input data {year}/constraints_equalities.csv
  input data {year}/constraints_inequalities.csv
  input data {year}/allocation_prior.csv
  input data {year}/README_inputs_{year}.md

All synthetic fields are documented.
"""

import os
import argparse
import numpy as np
import pandas as pd


# ---- knobs for synthetic generation ----
DEFAULT_SIGMA_FLOOR = 1.0      # absolute floor (same units as flows)
KAPPA_DEFAULT = 20.0           # Dirichlet concentration for allocation priors

# pedigree score ranges: 1(best) .. 4(worst)
PEDIGREE_P = {
    "coverage": (1, 3),
    "frequency": (1, 3),
    "spatial_boundary": (1, 2),
}

np.random.seed(42)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def compute_y_median(df: pd.DataFrame) -> pd.Series:
    vals = []
    for c in ["Value1", "Value2", "Value3", "Value4"]:
        if c in df.columns:
            vals.append(to_numeric(df[c]))
    if not vals:
        return pd.Series([np.nan] * len(df))
    Y = pd.concat(vals, axis=1)
    return Y.median(axis=1, skipna=True)


def classify_node(name: str) -> str:
    n = (name or "").lower()
    if "import" in n:
        return "import"
    if "export" in n:
        return "export"
    if "loss" in n or "waste" in n:
        return "loss"
    if "consumption" in n:
        return "consumption"
    return "process"


def main(year: int):
    input_dir = f"input data {year}"
    flow_file_year = os.path.join(input_dir, f"flow_data_{year}.csv")

    out_flow_canonical = os.path.join(input_dir, "flow_data.csv")
    out_nodes = os.path.join(input_dir, "nodes.csv")
    out_score = os.path.join(input_dir, "flow_data_score.csv")
    out_bounds = os.path.join(input_dir, "bounds.csv")
    out_eq = os.path.join(input_dir, "constraints_equalities.csv")
    out_ineq = os.path.join(input_dir, "constraints_inequalities.csv")
    out_alloc_prior = os.path.join(input_dir, "allocation_prior.csv")
    out_readme = os.path.join(input_dir, f"README_inputs_{year}.md")

    ensure_dir(input_dir)
    if not os.path.exists(flow_file_year):
        raise FileNotFoundError(flow_file_year)

    # Read with auto delimiter inference (handles CSV/TSV)
    df = pd.read_csv(flow_file_year, sep=None, engine="python")

    # Required columns check
    required = ["Flow index", "from_node_name", "to_node_name",
                "from_node_number", "to_node_number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{flow_file_year} missing columns: {missing}")

    # Coerce key columns
    for c in ["Flow index", "from_node_number", "to_node_number"]:
        df[c] = to_numeric(df[c]).astype("Int64")

    # Numeric replicate cols
    for c in ["Value1", "Value2", "Value3", "Value4"]:
        if c in df.columns:
            df[c] = to_numeric(df[c])

    df["y_median"] = compute_y_median(df)

    # 0) Write canonical flow_data.csv expected by loaders
    df.drop(columns=["y_median"], errors="ignore").to_csv(out_flow_canonical, index=False)

    # 1) nodes.csv
    nodes_from = df[["from_node_number", "from_node_name"]].rename(
        columns={"from_node_number": "node_id", "from_node_name": "node_name"}
    )
    nodes_to = df[["to_node_number", "to_node_name"]].rename(
        columns={"to_node_number": "node_id", "to_node_name": "node_name"}
    )
    nodes = pd.concat([nodes_from, nodes_to], ignore_index=True).dropna(subset=["node_id"])
    nodes["node_id"] = nodes["node_id"].astype(int)
    nodes = nodes.sort_values(["node_id"]).drop_duplicates("node_id", keep="first")
    nodes["node_type"] = nodes["node_name"].astype(str).apply(classify_node)
    nodes.to_csv(out_nodes, index=False)

    # 2) flow_data_score.csv (synthetic pedigree + obs_std)
    score = df[["Flow index", "from_node_name", "to_node_name", "y_median"]].copy()
    score["coverage"] = np.random.randint(PEDIGREE_P["coverage"][0], PEDIGREE_P["coverage"][1] + 1, size=len(score))
    score["frequency"] = np.random.randint(PEDIGREE_P["frequency"][0], PEDIGREE_P["frequency"][1] + 1, size=len(score))
    score["spatial_boundary"] = np.random.randint(PEDIGREE_P["spatial_boundary"][0], PEDIGREE_P["spatial_boundary"][1] + 1, size=len(score))

    # quality index q in [0,1]: higher = better
    avg = score[["coverage", "frequency", "spatial_boundary"]].mean(axis=1)
    q = (4.0 - avg) / 3.0
    q = np.clip(q, 0.0, 1.0)

    # map q to relative half-range r(q): best 5%, worst 60% (synthetic)
    r_min, r_max = 0.05, 0.60
    r = r_max - q * (r_max - r_min)

    score["obs_std"] = r * np.maximum(np.abs(score["y_median"].fillna(0.0)), DEFAULT_SIGMA_FLOOR)
    score = score.drop(columns=["y_median"])
    score.to_csv(out_score, index=False)

    # 3) bounds.csv (synthetic, var_idx = Flow index - 1)
    flow_index = df["Flow index"].astype(int).to_numpy()
    var_idx = flow_index - 1

    y = df["y_median"].to_numpy(dtype=float)
    have = np.isfinite(y)

    wide_ub = max(1.0, 10.0 * np.nanmax(np.abs(y[have]))) if np.any(have) else 1e6
    lower = np.where(have, 0.2 * y, 0.0)
    upper = np.where(have, 1.8 * y, wide_ub)

    bounds = pd.DataFrame({"var_idx": var_idx.astype(int), "lower": lower, "upper": upper})
    bounds.to_csv(out_bounds, index=False)

    # 4) allocation_prior.csv (Dirichlet priors from outgoing medians)
    tmp = df.copy()
    tmp["var_idx"] = tmp["Flow index"].astype(int) - 1
    tmp["y_median"] = tmp["y_median"].astype(float)

    recs = []
    for src, g in tmp.groupby("from_node_number"):
        g = g.copy()
        med = g["y_median"].to_numpy(dtype=float)
        if np.any(np.isfinite(med)) and np.nansum(med) > 0:
            shares = np.nan_to_num(med, nan=0.0)
            shares = shares / shares.sum()
        else:
            shares = np.ones(len(g)) / len(g)

        for row, share in zip(g.itertuples(index=False), shares):
            recs.append({
                "node_id": int(row.from_node_number),
                "target_node_id": int(row.to_node_number),
                "share_mean": float(share),
                "kappa": float(KAPPA_DEFAULT),
            })

    pd.DataFrame(recs).to_csv(out_alloc_prior, index=False)

    # 5) constraints_equalities.csv (minimal demo; update if you want real identities)
    # Include the same EOL identity only if those flow indices exist in this year's file.
    present_flow_ids = set(df["Flow index"].dropna().astype(int).tolist())
    eol_sector = list(range(234, 242))
    eol_out = [35, 36, 37]
    cid = 1
    eq_rows = []
    if all(fid in present_flow_ids for fid in (eol_sector + eol_out)):
        for fid in eol_sector:
            eq_rows.append({"constraint_id": cid, "var_idx": fid - 1, "coeff": 1.0, "rhs": 0.0})
        for fid in eol_out:
            eq_rows.append({"constraint_id": cid, "var_idx": fid - 1, "coeff": -1.0, "rhs": 0.0})

    eq = pd.DataFrame(eq_rows) if eq_rows else pd.DataFrame(
        columns=["constraint_id", "var_idx", "coeff", "rhs"]
    )
    eq.to_csv(out_eq, index=False)

    # 6) constraints_inequalities.csv (demo export upper bound)
    is_export_arc = df["to_node_name"].astype(str).str.lower().str.contains("export", na=False)
    export_var = (df.loc[is_export_arc, "Flow index"].astype(int) - 1).to_list()
    ineq_rows = []
    if export_var:
        export_total = np.nansum(df.loc[is_export_arc, "y_median"].to_numpy(dtype=float))
        rhs = float(2.0 * export_total) if np.isfinite(export_total) and export_total > 0 else float(wide_ub)
        cid = 1
        for v in export_var:
            ineq_rows.append({"constraint_id": cid, "var_idx": int(v), "coeff": 1.0, "rhs": rhs})

    ineq = pd.DataFrame(ineq_rows) if ineq_rows else pd.DataFrame(
        columns=["constraint_id", "var_idx", "coeff", "rhs"]
    )
    ineq.to_csv(out_ineq, index=False)

    # 7) README
    readme = f"""# Generated auxiliary inputs ({year}) (synthetic)

Generated from: `{flow_file_year}`

Wrote canonical file expected by loaders:
- `flow_data.csv` (copy of `flow_data_{year}.csv`)

Generated:
- nodes.csv: node list derived from from/to node numbers and names
- flow_data_score.csv: synthetic pedigree scores + obs_std (Gaussian noise)
- bounds.csv: synthetic bounds around median observations (0.2× to 1.8×); wide upper bounds otherwise
- constraints_equalities.csv: optional demo equality (only if relevant flow IDs exist)
- constraints_inequalities.csv: optional demo inequality (exports upper-bounded)
- allocation_prior.csv: Dirichlet allocation priors per source node derived from outgoing median flows

Notes:
- Replace synthetic constraints with engineering yield/capacity constraints as the model matures.
"""
    with open(out_readme, "w", encoding="utf-8") as f:
        f.write(readme)

    print("Wrote:")
    for p in [out_flow_canonical, out_nodes, out_score, out_bounds, out_eq, out_ineq, out_alloc_prior, out_readme]:
        print("  -", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()
    main(args.year)