"""
arcade_mfa_aluminum.graph
---------------------------
Node/flow graph utilities: mass-balance constraint construction, nullspace
computation, and posterior node-balance diagnostics. Shared by both vintages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MassBalanceSystem:
    A: np.ndarray          # (n_constraints, n_flows) balance matrix
    b: np.ndarray          # (n_constraints,) targets (zeros for pure balance)
    nodes: list            # node id per constraint row, same order as A rows


def build_node_balance(
    df: pd.DataFrame,
    n_flows: int,
    *,
    from_col: str = "from_node",
    to_col: str = "to_node",
    flow_idx_col: str = "flow_idx",
    min_degree: int = 1,
) -> MassBalanceSystem:
    """Build inflow-outflow=0 constraints for every internal node.

    An "internal" node is one that appears as both a from_node and a to_node
    within the supplied flow table (source/sink nodes are excluded).

    """
    nodes_from = set(pd.to_numeric(df[from_col], errors="coerce").dropna().astype(int))
    nodes_to = set(pd.to_numeric(df[to_col], errors="coerce").dropna().astype(int))
    internal = sorted(nodes_from & nodes_to)

    # Defensive check: every flow_idx must be a valid 0-based position in a
    # length-n_flows array. A bare IndexError here (as opposed to this
    # explicit message) almost always means an ingestion adapter left
    # flow_idx 1-based (or n_flows was computed inconsistently with the
    # actual flow_idx range) -- see load_aluminum_2017_from_workbook's
    # "-1" conversion for the fix pattern.
    idx_vals = pd.to_numeric(df[flow_idx_col], errors="coerce").dropna().astype(int)
    if len(idx_vals) and (idx_vals.min() < 0 or idx_vals.max() >= n_flows):
        raise ValueError(
            f"build_node_balance: {flow_idx_col} range is "
            f"[{idx_vals.min()}, {idx_vals.max()}] but n_flows={n_flows} "
            f"requires all values in [0, {n_flows - 1}]. This is almost always "
            f"a 1-based vs 0-based indexing mismatch in the upstream loader "
            f"(e.g. workbook flow_index is 1-based and wasn't converted)."
        )

    rows, nodes_used = [], []
    for node in internal:
        inflow_idx = df.loc[df[to_col] == node, flow_idx_col].to_numpy(dtype=int)
        outflow_idx = df.loc[df[from_col] == node, flow_idx_col].to_numpy(dtype=int)

        if len(inflow_idx) == 0 or len(outflow_idx) == 0:
            continue
        if len(inflow_idx) + len(outflow_idx) < max(2, min_degree):
            continue

        row = np.zeros(n_flows)
        row[inflow_idx] += 1.0
        row[outflow_idx] -= 1.0
        rows.append(row)
        nodes_used.append(node)

    A = np.vstack(rows) if rows else np.zeros((0, n_flows))
    b = np.zeros(A.shape[0])
    return MassBalanceSystem(A=A, b=b, nodes=nodes_used)


def nullspace(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for the nullspace of A (rows = constraints).

    Returns identity if A has no rows (unconstrained system).
    """
    if A.size == 0:
        n = A.shape[1] if A.ndim == 2 else 0
        return np.eye(n)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int((s > rtol * s.max()).sum())
    return Vt[rank:].T


def project_to_mass_balance(x: np.ndarray, A: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Least-squares projection of x onto {v : A v = 0}."""
    if A.shape[0] == 0:
        return x
    rhs = A @ x
    AAT = A @ A.T
    try:
        lam = np.linalg.solve(AAT + jitter * np.eye(AAT.shape[0]), rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(AAT + jitter * np.eye(AAT.shape[0]), rhs, rcond=None)[0]
    return x - A.T @ lam


def posterior_node_balance_diagnostics(
    samples: np.ndarray,   # (n_samples, n_flows)
    df: pd.DataFrame,
    *,
    from_col: str = "from_node",
    to_col: str = "to_node",
    flow_idx_col: str = "flow_idx",
) -> pd.DataFrame:
    """Per-node posterior residual (inflow - outflow) summary across draws.

    Takes a raw samples array, so it is independent of which sampler produced
    the draws.
    """
    all_nodes = sorted(
        set(pd.to_numeric(df[from_col], errors="coerce").dropna().astype(int))
        | set(pd.to_numeric(df[to_col], errors="coerce").dropna().astype(int))
    )
    records = []
    for node in all_nodes:
        inflow_idx = df.loc[df[to_col] == node, flow_idx_col].to_numpy(dtype=int)
        outflow_idx = df.loc[df[from_col] == node, flow_idx_col].to_numpy(dtype=int)
        if len(inflow_idx) == 0 or len(outflow_idx) == 0:
            continue

        inflow = samples[:, inflow_idx].sum(axis=1)
        outflow = samples[:, outflow_idx].sum(axis=1)
        residual = inflow - outflow
        throughput = np.maximum(np.maximum(inflow, outflow), 1e-9)
        residual_pct = np.abs(residual) / throughput * 100.0

        records.append({
            "node": node,
            "throughput_mean": float(np.mean(throughput)),
            # Residual in mass units and as a share of node throughput. Both are
            # reported because an absolute residual is only interpretable next
            # to the scale of the node it belongs to.
            "residual_mean": float(np.mean(residual)),
            "residual_q2.5": float(np.quantile(residual, 0.025)),
            "residual_q97.5": float(np.quantile(residual, 0.975)),
            "residual_pct_mean": float(np.mean(residual / throughput) * 100.0),
            "residual_pct_q2.5": float(np.quantile(residual / throughput, 0.025) * 100.0),
            "residual_pct_q97.5": float(np.quantile(residual / throughput, 0.975) * 100.0),
            "imbalance_pct_mean": float(np.mean(residual_pct)),
            "imbalance_pct_median": float(np.median(residual_pct)),
            "n_inflows": len(inflow_idx),
            "n_outflows": len(outflow_idx),
        })
    return pd.DataFrame(records)


def soft_mass_balance_block(
    mb: MassBalanceSystem, node_totals: dict, *, rel_sigma: float,
    min_total: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass balance as a penalized (soft) constraint block.

    Returns (A, b, sigma) for the rows of `mb` whose node carries meaningful
    throughput, with ``sigma_i = rel_sigma * T_i``. Scaling the tolerance by
    node throughput is what makes a single `rel_sigma` meaningful across a
    network spanning several orders of magnitude: a 10 kt node and a 10,000 kt
    node are then held to the same *relative* standard.

    Rows for nodes with negligible throughput are dropped -- a tolerance of
    zero would impose an effectively infinite penalty.
    """
    if rel_sigma <= 0:
        raise ValueError(f"mass_balance.rel_sigma must be positive, got {rel_sigma}")
    keep, sigmas = [], []
    for i, node in enumerate(mb.nodes):
        total = float(node_totals.get(int(node), 0.0))
        if total > min_total:
            keep.append(i)
            sigmas.append(rel_sigma * total)
    if not keep:
        return np.zeros((0, mb.A.shape[1])), np.zeros(0), np.zeros(0)
    return mb.A[keep], mb.b[keep], np.asarray(sigmas, dtype=float)
