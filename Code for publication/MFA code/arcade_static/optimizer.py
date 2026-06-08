"""
Generic optimization-based static MFA reconciliation.
Weighted least squares with exact mass-balance equality constraints.
"""

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize


def build_balance_matrix(df: pd.DataFrame) -> tuple:
    """Build mass-balance constraint matrix A such that A @ flows = 0."""
    nodes = sorted(set(df["from_node_number"]).union(df["to_node_number"]))
    intermediate = [
        n for n in nodes
        if (df["to_node_number"] == n).any() and (df["from_node_number"] == n).any()
    ]
    n_flows = len(df)
    A = np.zeros((len(intermediate), n_flows))
    for i, node in enumerate(intermediate):
        for j in range(n_flows):
            if df.loc[j, "to_node_number"] == node:
                A[i, j] = 1
            elif df.loc[j, "from_node_number"] == node:
                A[i, j] = -1
    return A, intermediate


def reconcile_optimization(
    df: pd.DataFrame,
    weighted_means: np.ndarray,
    uncertainties: np.ndarray,
    A_manual: np.ndarray = None,
    b_manual: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """
    Solve weighted-least-squares reconciliation with mass-balance constraints.

    Parameters
    ----------
    df : flow DataFrame with from_node_number, to_node_number
    weighted_means, uncertainties : per-flow observation summary
    A_manual, b_manual : optional additional linear equality constraints

    Returns
    -------
    dict with: x (reconciled flows), success, runtime, objective
    """
    n_flows = len(df)
    A_bal, intermediate = build_balance_matrix(df)

    # Combine balance + manual constraints
    if A_manual is not None and A_manual.shape[0] > 0:
        A_all = np.vstack([A_bal, A_manual])
        b_all = np.concatenate([np.zeros(A_bal.shape[0]), b_manual])
    else:
        A_all = A_bal
        b_all = np.zeros(A_bal.shape[0])

    if verbose:
        print(f"Flows: {n_flows}, balance constraints: {A_bal.shape[0]}, "
              f"manual constraints: {0 if A_manual is None else A_manual.shape[0]}")

    # Scale for numerical stability
    positive_means = weighted_means[weighted_means > 0]
    scale = float(np.median(positive_means)) if positive_means.size > 0 else 1.0

    x0 = (weighted_means / scale).copy()
    means_s = weighted_means / scale
    sigmas_s = uncertainties / scale
    b_all_s = b_all / scale

    def objective(x):
        return 0.5 * np.sum(((x - means_s) / sigmas_s) ** 2)

    def grad(x):
        return (x - means_s) / sigmas_s ** 2

    def constraint_fun(x):
        return A_all @ x - b_all_s

    constraints = {"type": "eq", "fun": constraint_fun, "jac": lambda x: A_all}
    bounds = [(0, None) for _ in range(n_flows)]

    t0 = time.time()
    result = minimize(
        objective, x0,
        method="trust-constr",
        jac=grad,
        constraints=constraints,
        bounds=bounds,
        options={"verbose": 1 if verbose else 0, "maxiter": 2000, "gtol": 1e-6},
    )
    runtime = time.time() - t0

    reconciled = result.x * scale
    return {
        "x": reconciled,
        "success": result.success,
        "runtime": runtime,
        "objective": float(result.fun),
        "intermediate_nodes": intermediate,
    }


def validate_mass_balance(df: pd.DataFrame, reconciled: np.ndarray) -> pd.DataFrame:
    """Compute per-node mass balance check."""
    records = []
    for node in sorted(set(df["from_node_number"]).union(df["to_node_number"])):
        inflow_idx  = df.loc[df["to_node_number"]   == node, "flow_idx"].to_numpy(int)
        outflow_idx = df.loc[df["from_node_number"] == node, "flow_idx"].to_numpy(int)
        if not len(inflow_idx) or not len(outflow_idx):
            continue
        in_sum  = float(reconciled[inflow_idx].sum())
        out_sum = float(reconciled[outflow_idx].sum())
        records.append({
            "node": node,
            "inflow": in_sum,
            "outflow": out_sum,
            "imbalance": in_sum - out_sum,
            "imbalance_pct": 100 * abs(in_sum - out_sum) / max(in_sum, out_sum, 1e-9),
        })
    return pd.DataFrame(records)