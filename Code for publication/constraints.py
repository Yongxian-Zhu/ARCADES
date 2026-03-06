#!/usr/bin/env python3
"""
constraints.py
Build the full constraint system  A x = b,  G x ≤ h,  ℓ ≤ x ≤ u
from the standardised ARCADE input tables.

Constraint sources
──────────────────
  • Mass balance at every internal node  (auto-generated from flow table)
  • Subset-sum / definitional identities (constraints_equalities.csv)
  • Yield and capacity bounds            (constraints_inequalities.csv)
  • Variable bounds                      (bounds.csv)
"""

import os
import numpy as np
import pandas as pd
from io_utils import load_csv, load_flows, normalize_name

MIN_POSITIVE = 1e-12


# ── mass-balance matrix (reused from core script, generalised) ──────
def build_mass_balance(df_flows: pd.DataFrame, n_vars: int):
    """Return  A_bal, b_bal, node_list  for  A_bal x = b_bal = 0."""
    from_col = "from_node_number"
    to_col = "to_node_number"
    from_all = df_flows[from_col].dropna().astype(int).unique()
    to_all = df_flows[to_col].dropna().astype(int).unique()
    internal = sorted(set(from_all).intersection(set(to_all)))

    A_rows, nodes_used = [], []
    for node in internal:
        in_idx = df_flows.loc[df_flows[to_col] == node, "var_idx"
                              ].dropna().astype(int).to_numpy()
        out_idx = df_flows.loc[df_flows[from_col] == node, "var_idx"
                               ].dropna().astype(int).to_numpy()
        if in_idx.size > 0 and out_idx.size > 0:
            row = np.zeros(n_vars)
            row[in_idx] += 1.0
            row[out_idx] -= 1.0
            A_rows.append(row)
            nodes_used.append(node)

    A = np.vstack(A_rows) if A_rows else np.zeros((0, n_vars))
    b = np.zeros(A.shape[0])
    return A, b, nodes_used


# ── additional equality constraints from CSV ────────────────────────
def load_equality_constraints(input_dir: str, n_vars: int):
    """Read constraints_equalities.csv → A_eq, b_eq.

    Expected columns: constraint_id, var_idx, coeff, rhs
    Each constraint_id defines one row of A_eq x = b_eq.
    """
    path = os.path.join(input_dir, "constraints_equalities.csv")
    if not os.path.exists(path):
        return np.zeros((0, n_vars)), np.zeros(0)

    df = load_csv(path, numeric_cols=["constraint_id", "var_idx",
                                       "coeff", "rhs"])
    cids = sorted(df["constraint_id"].dropna().unique())
    A_rows, b_vals = [], []
    for cid in cids:
        sub = df.loc[df["constraint_id"] == cid]
        row = np.zeros(n_vars)
        for _, r in sub.iterrows():
            idx = int(r["var_idx"])
            if 0 <= idx < n_vars:
                row[idx] = float(r["coeff"])
        rhs = sub["rhs"].dropna().iloc[0] if sub["rhs"].notna().any() else 0.0
        A_rows.append(row)
        b_vals.append(float(rhs))

    A = np.vstack(A_rows) if A_rows else np.zeros((0, n_vars))
    b = np.array(b_vals)
    return A, b


# ── inequality constraints from CSV ─────────────────────────────────
def load_inequality_constraints(input_dir: str, n_vars: int):
    """Read constraints_inequalities.csv → G, h  for  G x ≤ h.

    Expected columns: constraint_id, var_idx, coeff, rhs
    """
    path = os.path.join(input_dir, "constraints_inequalities.csv")
    if not os.path.exists(path):
        return np.zeros((0, n_vars)), np.zeros(0)

    df = load_csv(path, numeric_cols=["constraint_id", "var_idx",
                                       "coeff", "rhs"])
    cids = sorted(df["constraint_id"].dropna().unique())
    G_rows, h_vals = [], []
    for cid in cids:
        sub = df.loc[df["constraint_id"] == cid]
        row = np.zeros(n_vars)
        for _, r in sub.iterrows():
            idx = int(r["var_idx"])
            if 0 <= idx < n_vars:
                row[idx] = float(r["coeff"])
        rhs = sub["rhs"].dropna().iloc[0] if sub["rhs"].notna().any() else 0.0
        G_rows.append(row)
        h_vals.append(float(rhs))

    G = np.vstack(G_rows) if G_rows else np.zeros((0, n_vars))
    h = np.array(h_vals)
    return G, h


# ── variable bounds from CSV ────────────────────────────────────────
def load_bounds(input_dir: str, n_vars: int,
                default_lb=0.0, default_ub=np.inf):
    """Read bounds.csv → lb, ub arrays.

    Expected columns: var_idx, lower, upper
    """
    lb = np.full(n_vars, default_lb)
    ub = np.full(n_vars, default_ub)

    path = os.path.join(input_dir, "bounds.csv")
    if not os.path.exists(path):
        return lb, ub

    df = load_csv(path, numeric_cols=["var_idx", "lower", "upper"])
    for _, r in df.iterrows():
        idx = int(r["var_idx"])
        if 0 <= idx < n_vars:
            if pd.notna(r.get("lower")):
                lb[idx] = float(r["lower"])
            if pd.notna(r.get("upper")):
                ub[idx] = float(r["upper"])
    return lb, ub


# ── convenience: assemble everything ────────────────────────────────
def build_all_constraints(input_dir: str, df_flows: pd.DataFrame,
                          n_vars: int):
    """Return dict with keys A_eq, b_eq, G_ineq, h_ineq, lb, ub."""
    A_mb, b_mb, mb_nodes = build_mass_balance(df_flows, n_vars)
    A_extra, b_extra = load_equality_constraints(input_dir, n_vars)
    G, h = load_inequality_constraints(input_dir, n_vars)
    lb, ub = load_bounds(input_dir, n_vars)

    A_eq = np.vstack([A_mb, A_extra]) if A_extra.size else A_mb
    b_eq = np.concatenate([b_mb, b_extra]) if b_extra.size else b_mb

    print(f"Constraints assembled: {A_eq.shape[0]} equalities, "
          f"{G.shape[0]} inequalities, {n_vars} bounded variables")
    return dict(A_eq=A_eq, b_eq=b_eq,
                G_ineq=G, h_ineq=h,
                lb=lb, ub=ub,
                mb_nodes=mb_nodes)


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input data 2017")
    args = ap.parse_args()

    df = load_flows(args.input_dir)
    # assign contiguous var_idx
    pairs = df[["from_node_number", "to_node_number"]].drop_duplicates(
    ).reset_index(drop=True)
    pairs["var_idx"] = np.arange(len(pairs))
    df = df.merge(pairs, on=["from_node_number", "to_node_number"],
                  how="left")
    n = len(pairs)
    out = build_all_constraints(args.input_dir, df, n)
    print("A_eq shape:", out["A_eq"].shape)
    print("G_ineq shape:", out["G_ineq"].shape)