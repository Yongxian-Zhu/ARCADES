"""
Generic data ingestion and preprocessing for ARCADE static MFA reconciliation.
Works for any commodity (aluminum, steel, cement, fertilizer, pulp & paper, etc.).
"""

import pandas as pd
import numpy as np
import yaml
import re
from pathlib import Path


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_flows(path: str) -> pd.DataFrame:
    """
    Load flow observations from CSV.

    Required columns: from_node_number, to_node_number
    Optional columns: from_node_name, to_node_name, notes, upper_bound,
                      value_source1..N, quality_source1..N
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["from_node_number", "to_node_number"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")

    df["from_node_number"] = pd.to_numeric(df["from_node_number"], errors="coerce").astype(int)
    df["to_node_number"]   = pd.to_numeric(df["to_node_number"],   errors="coerce").astype(int)

    # Identify value and quality columns
    value_cols   = sorted([c for c in df.columns if c.startswith("value_source")])
    quality_cols = sorted([c for c in df.columns if c.startswith("quality_source")])

    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "upper_bound" in df.columns:
        df["upper_bound"] = pd.to_numeric(df["upper_bound"], errors="coerce")
    else:
        df["upper_bound"] = np.nan

    df["flow_idx"] = np.arange(len(df))
    return df, value_cols, quality_cols


def compute_weighted_observation(
    df: pd.DataFrame,
    value_cols: list,
    quality_cols: list,
    quality_to_sigma: dict,
    default_rel_sigma: float = 0.50,
) -> tuple:
    """
    Aggregate multi-source observations into a single weighted mean and uncertainty
    per flow. Returns (weighted_means, uncertainties) arrays.
    """
    n = len(df)
    means = np.zeros(n)
    uncerts = np.zeros(n)

    # Map qualitative scores to numeric weights
    def _quality_weight(q):
        if pd.isna(q):
            return 0.5
        if isinstance(q, str):
            qs = q.strip().lower()
            return {"high": 1.0, "medium": 0.6, "low": 0.2}.get(qs, 0.5)
        return float(q)

    def _quality_rel_sigma(q):
        if pd.isna(q):
            return default_rel_sigma
        if isinstance(q, str):
            return quality_to_sigma.get(q.strip().lower(), default_rel_sigma)
        # numeric score in [0,1] -> map linearly to sigma
        return default_rel_sigma * (1.0 - float(q)) + 0.05 * float(q)

    for i, row in df.iterrows():
        vals = [row[c] for c in value_cols if pd.notna(row[c]) and np.isfinite(row[c])]
        quals = []
        for vcol, qcol in zip(value_cols, quality_cols if quality_cols else []):
            if pd.notna(row[vcol]) and np.isfinite(row[vcol]):
                quals.append(row[qcol] if qcol in row else None)

        if vals:
            weights = np.array([_quality_weight(q) for q in quals])
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()

            means[i] = float(np.sum(np.array(vals) * weights))
            avg_rel_sigma = float(np.mean([_quality_rel_sigma(q) for q in quals]))
            uncerts[i] = max(avg_rel_sigma * means[i], 0.01 * means[i], 1.0)
        else:
            # No observation -> use upper bound center if available, else default
            ub = row.get("upper_bound", np.nan)
            if pd.notna(ub) and ub > 0:
                means[i] = 0.1 * ub
                uncerts[i] = 0.5 * ub
            else:
                means[i] = 0.0
                uncerts[i] = 100.0  # broad prior

    return means, uncerts


def parse_constraint_formula(expr: str, dest_node, valid_nodes: set) -> tuple:
    """
    Parse a linear constraint formula of the form:
        c1*node1 + c2*node2 - c3*node3 = target

    Returns (list of (coef, node_id, dest_node_id_or_None), target_value).
    """
    if pd.isna(expr) or "=" not in str(expr):
        return None, None

    expr = str(expr).replace(" ", "")
    lhs, rhs = expr.split("=", 1)
    rhs_is_constant = bool(re.match(r"^-?\d+\.?\d*$", rhs))

    def _parse_side(side, force_constants=False):
        side = side.replace("-", "+-")
        terms, const = [], 0.0
        for term in side.split("+"):
            if not term:
                continue
            m = re.match(r"^(?:(-?\d*\.?\d+)\*)?(-?\d+)(?:/(-?\d*\.?\d+))?$", term)
            if m:
                coef_str, node_str, div_str = m.groups()
                coef = 1.0 if coef_str is None else (-1.0 if coef_str == "-" else float(coef_str))
                node_val = int(node_str)
                if node_val < 0:
                    coef, node_val = -abs(coef), abs(node_val)
                if div_str:
                    coef /= float(div_str)
                if force_constants or (node_val not in valid_nodes):
                    const += coef * node_val
                else:
                    terms.append((coef, node_val))
            else:
                try:
                    const += float(term)
                except ValueError:
                    return None, None
        return terms, const

    lhs_terms, lhs_const = _parse_side(lhs, force_constants=False)
    rhs_terms, rhs_const = _parse_side(rhs, force_constants=rhs_is_constant)

    if lhs_terms is None or rhs_terms is None:
        return None, None

    combined = lhs_terms + [(-c, n) for c, n in rhs_terms]
    target = rhs_const - lhs_const

    try:
        dest = int(dest_node) if pd.notna(dest_node) else None
    except (ValueError, TypeError):
        dest = None

    return [(c, n, dest) for (c, n) in combined], target