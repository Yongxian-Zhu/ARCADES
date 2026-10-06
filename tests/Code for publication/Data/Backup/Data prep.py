#!/usr/bin/env python3
"""
Generate ARCADE auxiliary input tables for 2017 from flow_data.csv.

Inputs:
  input data 2017/flow_data.csv   (your example)

Outputs (created/overwritten):
  input data 2017/nodes.csv
  input data 2017/flow_data_score.csv
  input data 2017/bounds.csv
  input data 2017/constraints_equalities.csv
  input data 2017/constraints_inequalities.csv
  input data 2017/allocation_prior.csv
  input data 2017/README_inputs_2017.md

All synthetic fields are clearly documented.
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = "input data 2017"
FLOW_FILE = os.path.join(INPUT_DIR, "flow_data.csv")

OUT_NODES = os.path.join(INPUT_DIR, "nodes.csv")
OUT_SCORE = os.path.join(INPUT_DIR, "flow_data_score.csv")
OUT_BOUNDS = os.path.join(INPUT_DIR, "bounds.csv")
OUT_EQ = os.path.join(INPUT_DIR, "constraints_equalities.csv")
OUT_INEQ = os.path.join(INPUT_DIR, "constraints_inequalities.csv")
OUT_ALLOC_PRIOR = os.path.join(INPUT_DIR, "allocation_prior.csv")
OUT_README = os.path.join(INPUT_DIR, "README_inputs_2017.md")

# ---- knobs for synthetic generation ----
DEFAULT_REL_SIGMA = 0.10       # if no pedigree: 10% relative std
DEFAULT_SIGMA_FLOOR = 1.0      # absolute floor (same units as flows)
KAPPA_DEFAULT = 20.0           # Dirichlet concentration for allocation priors

# pedigree score ranges: 1(best) .. 4(worst)
PEDIGREE_P = {
    "coverage": (1, 3),
    "frequency": (1, 3),
    "spatial_boundary": (1, 2),
}

np.random.seed(42)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return s

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def compute_y_median(df):
    vals = []
    for c in ["Value1", "Value2", "Value3", "Value4"]:
        if c in df.columns:
            vals.append(to_numeric(df[c]))
    Y = pd.concat(vals, axis=1)
    return Y.median(axis=1, skipna=True)

def classify_node(name):
    n = name.lower()
    if "import" in n:
        return "import"
    if "export" in n:
        return "export"
    if "loss" in n or "waste" in n:
        return "loss"
    if "consumption" in n:
        return "consumption"
    return "process"

def main():
    ensure_dir(INPUT_DIR)
    if not os.path.exists(FLOW_FILE):
        raise FileNotFoundError(FLOW_FILE)

    df = pd.read_csv(FLOW_FILE, sep=None, engine="python")
    # standardize columns
    for c in ["from_node_name", "to_node_name"]:
        df[c] = df[c].apply(normalize_name)
    for c in ["Flow index", "from_node_number", "to_node_number"]:
        df[c] = to_numeric(df[c]).astype("Int64")

    y_med = compute_y_median(df)
    df["y_median"] = y_med

    # -------------------------
    # 1) nodes.csv
    # -------------------------
    nodes_from = df[["from_node_number", "from_node_name"]].rename(
        columns={"from_node_number":"node_id", "from_node_name":"node_name"}
    )
    nodes_to = df[["to_node_number", "to_node_name"]].rename(
        columns={"to_node_number":"node_id", "to_node_name":"node_name"}
    )
    nodes = pd.concat([nodes_from, nodes_to], ignore_index=True).dropna(subset=["node_id"])
    nodes["node_id"] = nodes["node_id"].astype(int)
    # if same id appears with slightly different names, keep first
    nodes = nodes.sort_values(["node_id"]).drop_duplicates("node_id", keep="first")
    nodes["node_type"] = nodes["node_name"].apply(classify_node)
    nodes.to_csv(OUT_NODES, index=False)

    # -------------------------
    # 2) flow_data_score.csv (synthetic)
    # -------------------------
    score = df[["Flow index", "from_node_name", "to_node_name", "y_median"]].copy()
    score["coverage"] = np.random.randint(PEDIGREE_P["coverage"][0], PEDIGREE_P["coverage"][1]+1, size=len(score))
    score["frequency"] = np.random.randint(PEDIGREE_P["frequency"][0], PEDIGREE_P["frequency"][1]+1, size=len(score))
    score["spatial_boundary"] = np.random.randint(PEDIGREE_P["spatial_boundary"][0], PEDIGREE_P["spatial_boundary"][1]+1, size=len(score))

    # quality index q in [0,1] like the SI rubric: higher = better
    avg = score[["coverage","frequency","spatial_boundary"]].mean(axis=1)
    q = (4.0 - avg) / 3.0
    q = np.clip(q, 0.0, 1.0)

    # map q to relative half-range r(q): best 5%, worst 60% (synthetic)
    r_min, r_max = 0.05, 0.60
    r = r_max - q*(r_max - r_min)

    # obs_std = r * max(|y|, floor)
    score["obs_std"] = r * np.maximum(np.abs(score["y_median"].fillna(0.0)), DEFAULT_SIGMA_FLOOR)

    score = score.drop(columns=["y_median"])
    score.to_csv(OUT_SCORE, index=False)

    # -------------------------
    # 3) bounds.csv (synthetic, consistent with your core script style)
    # var_idx is 0-based index aligned to Flow index (assuming 1..N)
    # -------------------------
    flow_index = df["Flow index"].astype(int).to_numpy()
    var_idx = flow_index - 1

    # compute per-flow bounds around median observation where available
    y = df["y_median"].to_numpy(dtype=float)
    have = np.isfinite(y)

    # wide upper bound for unobserved
    wide_ub = max(1.0, 10.0*np.nanmax(np.abs(y[have]))) if np.any(have) else 1e6
    lower = np.where(have, 0.2*y, 0.0)
    upper = np.where(have, 1.8*y, wide_ub)

    bounds = pd.DataFrame({
        "var_idx": var_idx.astype(int),
        "lower": lower,
        "upper": upper,
    })
    bounds.to_csv(OUT_BOUNDS, index=False)

    # -------------------------
    # 4) allocation_prior.csv (synthetic-from-data)
    # For each source node, normalize outgoing median flows to shares.
    # -------------------------
    tmp = df.copy()
    tmp["var_idx"] = (tmp["Flow index"].astype(int) - 1)
    tmp["y_median"] = tmp["y_median"].astype(float)

    # group by source node
    recs = []
    for src, g in tmp.groupby("from_node_number"):
        g = g.copy()
        # use only arcs with finite median; if none, uniform shares
        med = g["y_median"].to_numpy(dtype=float)
        if np.any(np.isfinite(med)) and np.nansum(med) > 0:
            shares = np.nan_to_num(med, nan=0.0)
            shares = shares / shares.sum()
        else:
            shares = np.ones(len(g)) / len(g)

        for (row, share) in zip(g.itertuples(index=False), shares):
            recs.append({
                "node_id": int(row.from_node_number),
                "target_node_id": int(row.to_node_number),
                "share_mean": float(share),
                "kappa": float(KAPPA_DEFAULT),
            })

    alloc_prior = pd.DataFrame(recs)
    alloc_prior.to_csv(OUT_ALLOC_PRIOR, index=False)

    # -------------------------
    # 5) constraints_equalities.csv (minimal synthetic example)
    # One example constraint: (sum sectoral EOL) - (EOL export+loss+recycled) = 0
    # Flow indices:
    #   sectoral EOL: 234..241
    #   EOL outflows: 35,36,37  (from node 18 to 93/94/19)
    # Note: This is only a demo identity; your model can run without it.
    # -------------------------
    eol_sector = list(range(234, 242))  # inclusive
    eol_out = [35, 36, 37]
    cid = 1
    eq_rows = []
    for fid in eol_sector:
        eq_rows.append({"constraint_id": cid, "var_idx": fid-1, "coeff": 1.0, "rhs": 0.0})
    for fid in eol_out:
        eq_rows.append({"constraint_id": cid, "var_idx": fid-1, "coeff": -1.0, "rhs": 0.0})

    eq = pd.DataFrame(eq_rows)
    eq.to_csv(OUT_EQ, index=False)

    # -------------------------
    # 6) constraints_inequalities.csv (minimal synthetic)
    # Example: constrain total "export" arcs (by target node type) to be <= 2x observed total.
    # We'll identify export arcs by target node name containing "Export".
    # -------------------------
    is_export_arc = df["to_node_name"].str.lower().str.contains("export", na=False)
    export_var = (df.loc[is_export_arc, "Flow index"].astype(int) - 1).to_list()

    ineq_rows = []
    if export_var:
        export_total = np.nansum(df.loc[is_export_arc, "y_median"].to_numpy(dtype=float))
        rhs = float(2.0 * export_total) if np.isfinite(export_total) and export_total > 0 else float(wide_ub)
        cid = 1
        for v in export_var:
            ineq_rows.append({"constraint_id": cid, "var_idx": int(v), "coeff": 1.0, "rhs": rhs})

    ineq = pd.DataFrame(ineq_rows) if ineq_rows else pd.DataFrame(
        columns=["constraint_id","var_idx","coeff","rhs"]
    )
    ineq.to_csv(OUT_INEQ, index=False)

    # -------------------------
    # 7) README
    # -------------------------
    readme = f"""# Generated 2017 auxiliary inputs (synthetic)

These files were generated from: `{FLOW_FILE}`

Generated:
- nodes.csv: node list derived from from/to node numbers and names
- flow_data_score.csv: synthetic pedigree scores + obs_std (Gaussian noise)
- bounds.csv: synthetic bounds around median observations (0.2× to 1.8×); wide upper bounds otherwise
- constraints_equalities.csv: one demo equality (sectoral EOL sum vs EOL outflows)
- constraints_inequalities.csv: one demo inequality (total export arcs upper-bounded)
- allocation_prior.csv: Dirichlet allocation priors per source node derived from outgoing median flows

Notes:
- Your core PyMC flow-based script can run without most of these files (only flow_data.csv is required).
- The node–allocation inference uses allocation_prior.csv if present; otherwise it falls back to weak priors.
- Replace these synthetic constraints with engineering constraints (yields/capacity) as you formalize the model.
"""
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write(readme)

    print("Wrote:")
    for p in [OUT_NODES, OUT_SCORE, OUT_BOUNDS, OUT_EQ, OUT_INEQ, OUT_ALLOC_PRIOR, OUT_README]:
        print("  -", p)

if __name__ == "__main__":
    main()