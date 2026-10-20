"""
arcade_mfa_aluminum.allocation
------------------------------
Transfer node allocation ratios from the 2017 posterior into the 2022
reconciliation as *soft* constraints.

Motivation
----------
The 2022 reconciliation is strongly under-determined: only about a fifth of
flows carry an observation. The remainder are pinned by mass balance and by the
bound heuristic alone, and many are effectively unidentified, with posterior
spread approaching that of a uniform distribution over the feasible range.

The 2017 posterior holds information the 2022 data lacks: how each node splits
its throughput. Such splits are technological and largely persistent, so
carrying them forward supplies the missing structure -- provided they are
imposed as a relaxed preference rather than as fact, since real change occurs
between vintages.

The constraint is exactly linear
--------------------------------
For node p with outflow set O(p) and 2017 share s_j::

    x_j / sum_{k in O(p)} x_k = s_j              (nonlinear in x)
    <=>  x_j - s_j * sum_{k in O(p)} x_k = 0     (LINEAR: s_j is a constant)

so row j of R is::

    R[j, j] = 1 - s_j
    R[j, k] = -s_j     for k in O(p), k != j

Rows belonging to one node sum to the zero vector (because sum_j s_j = 1), so a
node of degree d contributes rank d-1. All d rows are kept: as a quadratic
penalty that is a symmetric, well-defined form, not an over-constraint.

Applied softly, as a Gaussian penalty on top of the existing quadratic
objective::

    + 0.5 * (R x)^T W (R x),    W = diag(1 / tau^2)
    =>  Q += R^T W R

Mass balance and the box bounds stay hard; this is the relaxed layer. Since
R^T W R is positive semi-definite, Q stays PSD, and because `pipeline.run`
hands `map_result.Q` to the sampler, the information propagates into the
posterior draws with no change to the sampler itself.

Setting tau
-----------
The row residual `x_j - s_j * T_p` is in flow units, where T_p is the node's
total outflow. Since `residual_j = T_p * (w_j - s_j)`::

    tau_j = clip(relaxation * sigma_2017_j, min_share_sigma, max_share_sigma) * T_ref_p

`relaxation` is the adjustable knob: 1.0 treats the 2017 ratio as about as
reliable for 2022 as it was for 2017; larger values loosen it. The floor stops
a razor-tight 2017 share (remelting's 0.0092) from effectively becoming a hard
constraint on a different year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.paths import long_path, prepare_output


@dataclass
class AllocationShares:
    """Per-flow allocation shares estimated from a source-year posterior.

    One entry per (branching node, outflow) pair; nodes with a single outflow
    carry no ratio information and are excluded.
    """
    node: np.ndarray          # (n_rows,) source node id
    flow_idx: np.ndarray      # (n_rows,) flow index (0-based canonical position)
    share_mean: np.ndarray    # (n_rows,) posterior mean of x_j / node total
    share_sd: np.ndarray      # (n_rows,) posterior sd of that share
    n_out: np.ndarray         # (n_rows,) number of outflows at the node
    source_year: int

    def __len__(self) -> int:
        return len(self.flow_idx)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "node": self.node,
            "flow_idx": self.flow_idx,
            "share_mean": self.share_mean,
            "share_sd": self.share_sd,
            "n_out": self.n_out,
            "source_year": self.source_year,
        })


def compute_allocation_shares(
    samples: np.ndarray,
    df: pd.DataFrame,
    *,
    from_col: str = "from_node",
    flow_idx_col: str = "flow_idx",
    min_outflows: int = 2,
    min_node_total: float = 1e-9,
    min_valid_draws: int = 50,
) -> tuple[AllocationShares, np.ndarray]:
    """Estimate allocation shares from posterior draws.

    Parameters
    ----------
    samples : (n_draws, n_flows) posterior draws in flow space.
    df : canonical flow table (needs `from_node` and `flow_idx`).

    Returns
    -------
    (AllocationShares, w_samples) where `w_samples` is (n_draws, n_flows) with
    per-draw shares and NaN for flows that belong to no branching node.

    Shares are computed per draw and only then summarized, so `share_sd` is a
    genuine posterior spread rather than a propagated point estimate.
    """
    samples = np.asarray(samples, dtype=float)
    n_draws, n_flows = samples.shape

    w_samples = np.full((n_draws, n_flows), np.nan)
    nodes, flows, means, sds, degrees = [], [], [], [], []

    for node, sub in df.groupby(from_col):
        idx = pd.to_numeric(sub[flow_idx_col], errors="coerce").dropna().astype(int).to_numpy()
        if len(idx) < min_outflows:
            continue

        total = samples[:, idx].sum(axis=1)
        ok = total > min_node_total
        if int(ok.sum()) < min_valid_draws:
            # Node carries essentially no throughput in the source year, so the
            # split is undefined. Skip rather than emit a NaN-poisoned row.
            continue

        w = samples[np.ix_(np.where(ok)[0], idx)] / total[ok, None]
        w_samples[np.ix_(np.where(ok)[0], idx)] = w

        for j, fi in enumerate(idx):
            nodes.append(int(node))
            flows.append(int(fi))
            means.append(float(w[:, j].mean()))
            sds.append(float(w[:, j].std(ddof=1)))
            degrees.append(len(idx))

    shares = AllocationShares(
        node=np.asarray(nodes, dtype=int),
        flow_idx=np.asarray(flows, dtype=int),
        share_mean=np.asarray(means, dtype=float),
        share_sd=np.asarray(sds, dtype=float),
        n_out=np.asarray(degrees, dtype=int),
        source_year=int(df["year"].iloc[0]) if "year" in df.columns and len(df) else 0,
    )
    return shares, w_samples


def save_allocation_shares(shares: AllocationShares, path: str) -> None:
    """Persist shares to .npz (mirrors transfer.save_prior_npy)."""
    np.savez(
        prepare_output(path),
        node=shares.node,
        flow_idx=shares.flow_idx,
        share_mean=shares.share_mean,
        share_sd=shares.share_sd,
        n_out=shares.n_out,
        source_year=np.array([shares.source_year]),
    )


def load_allocation_shares(path: str) -> AllocationShares:
    with np.load(long_path(path)) as z:
        return AllocationShares(
            node=z["node"], flow_idx=z["flow_idx"],
            share_mean=z["share_mean"], share_sd=z["share_sd"],
            n_out=z["n_out"], source_year=int(z["source_year"][0]),
        )


@dataclass
class AllocationConstraints:
    """Soft linear constraint system R x ~ 0 with per-row tolerance `tau`."""
    R: np.ndarray             # (n_rows, n_flows)
    tau: np.ndarray           # (n_rows,) sd of each row residual, in flow units
    meta: pd.DataFrame        # audit trail: node, flow, share, sigmas, tau

    @property
    def n_rows(self) -> int:
        return self.R.shape[0]

    @property
    def n_nodes(self) -> int:
        return int(self.meta["node"].nunique()) if len(self.meta) else 0


def node_totals_from_flows(
    x: np.ndarray, df: pd.DataFrame, *, from_col: str = "from_node",
    flow_idx_col: str = "flow_idx",
) -> dict:
    """Total outflow per source node for a given flow vector."""
    out = {}
    for node, sub in df.groupby(from_col):
        idx = pd.to_numeric(sub[flow_idx_col], errors="coerce").dropna().astype(int).to_numpy()
        out[int(node)] = float(np.abs(x[idx]).sum())
    return out


def build_allocation_constraints(
    df: pd.DataFrame,
    shares: AllocationShares,
    node_totals: dict,
    *,
    n_flows: int,
    relaxation: float = 2.0,
    min_share_sigma: float = 0.02,
    max_share_sigma: float = 0.5,
    exclude_nodes: tuple = (),
    min_node_total: float = 1e-6,
    from_col: str = "from_node",
    flow_idx_col: str = "flow_idx",
) -> AllocationConstraints:
    """Build the soft allocation constraint system for a target year.

    `node_totals` maps node id -> reference total outflow in the TARGET year,
    used to convert a share-space tolerance into flow units. Supplying it from
    a MAP warm-up keeps the linearization self-consistent with the target
    year's own magnitudes instead of inheriting the source year's.

    Nodes present in `shares` but absent (or with ~zero throughput) in the
    target graph are skipped.
    """
    if relaxation <= 0:
        raise ValueError(f"relaxation must be positive, got {relaxation}")
    # Shares are addressed BY flow_idx. If the flow set has changed since they
    # were exported (nodes removed => flow_idx renumbered), those indices now
    # name different flows and the constraints would be silently wrong rather
    # than obviously broken. Refuse instead.
    if len(shares) and int(shares.flow_idx.max()) >= n_flows:
        raise ValueError(
            f"Allocation shares reference flow_idx up to "
            f"{int(shares.flow_idx.max())} but this run has only {n_flows} flows. "
            f"The shares were built for a different flow set; re-run the source-year "
            f"config to regenerate them."
        )
    if min_share_sigma > max_share_sigma:
        raise ValueError(
            f"min_share_sigma ({min_share_sigma}) exceeds max_share_sigma ({max_share_sigma})"
        )

    # Target-year outflow membership, which may differ from the source year.
    node_out = {
        int(node): pd.to_numeric(sub[flow_idx_col], errors="coerce").dropna().astype(int).to_numpy()
        for node, sub in df.groupby(from_col)
    }
    exclude = {int(n) for n in exclude_nodes}
    by_node: dict[int, list[int]] = {}
    for row, node in enumerate(shares.node):
        by_node.setdefault(int(node), []).append(row)

    rows, taus, recs = [], [], []
    for node, row_ids in sorted(by_node.items()):
        if node in exclude or node not in node_out:
            continue
        idx = node_out[node]
        total = float(node_totals.get(node, 0.0))
        if total <= min_node_total or len(idx) < 2:
            continue

        share_rows = {int(shares.flow_idx[r]): r for r in row_ids}
        if not set(idx).issubset(share_rows):
            # Topology changed between vintages; skip rather than guess.
            continue

        # Renormalize in case the source-year node had extra//missing edges.
        s = np.array([shares.share_mean[share_rows[int(f)]] for f in idx], dtype=float)
        s_sum = s.sum()
        if not np.isfinite(s_sum) or s_sum <= 0:
            continue
        s = s / s_sum
        sd = np.array([shares.share_sd[share_rows[int(f)]] for f in idx], dtype=float)

        for j, f in enumerate(idx):
            row = np.zeros(n_flows)
            row[idx] = -s[j]
            row[f] += 1.0          # +=, so the diagonal becomes 1 - s_j
            share_sigma = float(np.clip(relaxation * sd[j], min_share_sigma, max_share_sigma))
            rows.append(row)
            taus.append(share_sigma * total)
            recs.append({
                "node": node, "flow_idx": int(f), "n_out": len(idx),
                "target_share": s[j], "source_share_sd": sd[j],
                "share_sigma": share_sigma, "node_total_ref": total,
                "tau_flow_units": share_sigma * total,
            })

    if not rows:
        return AllocationConstraints(
            R=np.zeros((0, n_flows)), tau=np.zeros(0), meta=pd.DataFrame()
        )
    return AllocationConstraints(
        R=np.vstack(rows), tau=np.asarray(taus, dtype=float), meta=pd.DataFrame(recs)
    )


def allocation_residual_report(
    samples: np.ndarray, constraints: AllocationConstraints,
) -> pd.DataFrame:
    """Realized allocation deviation per row, in SHARE units.

    Answers whether the target year actually honoured the transferred ratios or
    fought them: `share_dev_mean` far outside `share_sigma` means the target
    year's own data disagrees with the source-year split, which is a finding in
    its own right rather than a defect.
    """
    if constraints.n_rows == 0:
        return pd.DataFrame()

    resid = samples @ constraints.R.T                 # (n_draws, n_rows), flow units
    scale = np.maximum(constraints.meta["node_total_ref"].to_numpy(), 1e-12)
    share_dev = resid / scale[None, :]

    out = constraints.meta.copy()
    out["share_dev_mean"] = share_dev.mean(axis=0)
    out["share_dev_sd"] = share_dev.std(axis=0)
    out["share_dev_abs_mean"] = np.abs(share_dev).mean(axis=0)
    out["z_vs_tolerance"] = out["share_dev_mean"] / np.maximum(out["share_sigma"], 1e-12)
    return out
