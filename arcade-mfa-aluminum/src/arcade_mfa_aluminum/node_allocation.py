"""
arcade_mfa_aluminum.node_allocation
-----------------------------------
Node-allocation formulation: node throughputs and allocation shares are the
variables, and flows are recovered from them.

Why a second formulation
------------------------
The flow-based formulation treats the flow magnitudes themselves as the
variables. This one treats each source node's throughput ``T_p`` and the shares
``s_{p,j}`` that split it, with::

    x_j = T_p * s_{p,j},    p = from_node(j),    sum_{j in O(p)} s_{p,j} = 1

The two are reparameterisations of the same system -- 80 node totals plus 234
shares under 80 simplex equalities leaves 234 degrees of freedom, exactly the
flow count -- so agreement between them is a genuine check on the reconciliation
rather than a restatement of it.

The formulation also gives the pedigree score somewhere natural to act on the
*split* rather than on individual flows. Equation S17 maps a node's quality
index to a Dirichlet concentration::

    kappa_p = kappa_min + q_p (kappa_max - kappa_min)

so a well-evidenced node has a tight split and a poorly-evidenced one is free to
move. That is a statement about allocation, which the flow formulation cannot
express directly.

Converting flow data to node totals and shares
----------------------------------------------
Per source node, from the per-flow observations:

*All outflows observed* -- the node total is their sum, with sigma combined in
quadrature, and the observed shares are the normalised values. The node
contributes both a normal likelihood on ``T_p`` and a Dirichlet likelihood on
``s_p``.

*Some outflows observed* -- no total and no Dirichlet, because a partial sum is
not the node total and a partial split is not a simplex. The per-flow normal
likelihood on ``x_j = T_p s_{p,j}`` is retained instead, so **no observation is
discarded**. This case dominates the sparser target year.

*None observed* -- carried entirely by mass balance and the transferred prior.

Quality index
-------------
S15 is ``r(q) = rmax - q (rmax - rmin)``, so a relative sigma inverts to a
quality index by ``q = (rmax - r) / (rmax - rmin)``. The relative sigma used is
the one the likelihood actually applies -- combined across sources by precision
and after any per-flow override -- rather than a fresh reading of the pedigree
columns, so a customs-tracked flow is correctly treated as well evidenced.

Node scores are the unweighted mean of their outflows' quality indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Endpoints of the S15 mapping. These mirror the defaults of
#: `adapters.aluminum.adapter.pedigree_to_rel_sigma` (sigma_rel_best /
#: sigma_rel_worst); a config that changes those must change these too or the
#: quality index will not invert consistently.
DEFAULT_R_MIN = 0.05
DEFAULT_R_MAX = 1.00

#: Dirichlet concentration endpoints for S17.
DEFAULT_KAPPA_MIN = 2.0
DEFAULT_KAPPA_MAX = 200.0


@dataclass
class NodeAllocationLayout:
    """Index bookkeeping between (node total, share) space and flow space.

    `nodes` lists every node with at least one outflow, in ascending id order.
    `out_flows[i]` holds the 0-based flow indices leaving `nodes[i]`, and
    `flow_node_pos[j]` is the position in `nodes` of flow j's source, so that
    expanding to flows is a single gather.
    """

    nodes: np.ndarray
    out_flows: list
    flow_node_pos: np.ndarray
    n_flows: int

    @property
    def n_nodes(self) -> int:
        return int(len(self.nodes))

    @property
    def branching(self) -> np.ndarray:
        """Positions of nodes with more than one outflow -- the only ones whose
        split carries information."""
        return np.array([i for i, o in enumerate(self.out_flows) if len(o) > 1], dtype=int)

    def simplex_matrix(self) -> np.ndarray:
        """(n_nodes, n_flows) indicator: row i sums the shares of node i."""
        S = np.zeros((self.n_nodes, self.n_flows))
        for i, idx in enumerate(self.out_flows):
            S[i, idx] = 1.0
        return S

    def expand(self, totals: np.ndarray, shares: np.ndarray) -> np.ndarray:
        """x_j = T_{from(j)} * s_j."""
        return np.asarray(totals, dtype=float)[self.flow_node_pos] * np.asarray(shares, dtype=float)


@dataclass
class NodeAllocationData:
    """Observations mapped from flow space into node-allocation space."""

    layout: NodeAllocationLayout
    total_value: np.ndarray          # (n_nodes,) NaN where not fully observed
    total_sigma: np.ndarray          # (n_nodes,) NaN where not fully observed
    share_obs: np.ndarray            # (n_flows,) NaN where the node is not fully observed
    kappa: np.ndarray                # (n_nodes,) Dirichlet concentration, S17
    quality: np.ndarray              # (n_nodes,) q_p in [0, 1]
    resid_flow_idx: np.ndarray       # observed flows at partially observed nodes
    resid_value: np.ndarray
    resid_sigma: np.ndarray
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_total_obs(self) -> int:
        return int(np.isfinite(self.total_value).sum())

    @property
    def n_share_obs(self) -> int:
        return int(np.isfinite(self.share_obs).sum())


def build_layout(df: pd.DataFrame, n_flows: int, *,
                 from_col: str = "from_node", flow_idx_col: str = "flow_idx") -> NodeAllocationLayout:
    """Group flows by source node.

    Every flow belongs to exactly one source node, so the shares vector has the
    same length as the flow vector and `flow_node_pos` is a total map.
    """
    src = pd.to_numeric(df[from_col], errors="coerce")
    idx = pd.to_numeric(df[flow_idx_col], errors="coerce")
    if idx.isna().any() or src.isna().any():
        raise ValueError("build_layout: from_node and flow_idx must be non-null")
    idx = idx.astype(int).to_numpy()
    src = src.astype(int).to_numpy()
    if idx.min() < 0 or idx.max() >= n_flows:
        raise ValueError(
            f"build_layout: flow_idx spans [{idx.min()}, {idx.max()}] which is outside "
            f"[0, {n_flows - 1}]. The canonical loader emits 0-based indices."
        )

    nodes = np.array(sorted(set(src.tolist())), dtype=int)
    pos_of = {int(nd): i for i, nd in enumerate(nodes)}
    out_flows = [np.sort(idx[src == nd]) for nd in nodes]

    flow_node_pos = np.full(n_flows, -1, dtype=int)
    flow_node_pos[idx] = [pos_of[int(s)] for s in src]
    if (flow_node_pos < 0).any():
        missing = int((flow_node_pos < 0).sum())
        raise ValueError(f"build_layout: {missing} flow(s) have no source node assignment")

    return NodeAllocationLayout(nodes=nodes, out_flows=out_flows,
                                flow_node_pos=flow_node_pos, n_flows=n_flows)


def combine_observations(obs: pd.DataFrame, n_flows: int) -> tuple:
    """Collapse the long-form observation table to one value and sigma per flow.

    Sources on the same flow combine by precision, matching how `solve_map`
    treats them: the value is the inverse-variance weighted mean and the sigma
    is that of the combined estimate. Unobserved flows are NaN.
    """
    y = np.full(n_flows, np.nan)
    sg = np.full(n_flows, np.nan)
    if obs is None or not len(obs):
        return y, sg
    idx = obs["flow_idx"].to_numpy(dtype=int)
    val = obs["value"].to_numpy(dtype=float)
    sig = obs["sigma"].to_numpy(dtype=float)
    w = 1.0 / np.maximum(sig ** 2, 1e-12)
    num = np.zeros(n_flows)
    den = np.zeros(n_flows)
    np.add.at(num, idx, w * val)
    np.add.at(den, idx, w)
    seen = den > 0
    y[seen] = num[seen] / den[seen]
    sg[seen] = np.sqrt(1.0 / den[seen])
    return y, sg


def quality_index_from_rel_sigma(rel_sigma, *, r_min: float = DEFAULT_R_MIN,
                                 r_max: float = DEFAULT_R_MAX) -> np.ndarray:
    """Invert S15: q = (rmax - r) / (rmax - rmin), clipped to [0, 1].

    r <= rmin gives q = 1 (best evidenced); r >= rmax gives q = 0.
    """
    if r_max <= r_min:
        raise ValueError(f"r_max ({r_max}) must exceed r_min ({r_min})")
    r = np.asarray(rel_sigma, dtype=float)
    q = (r_max - r) / (r_max - r_min)
    return np.clip(q, 0.0, 1.0)


def pedigree_to_concentration(quality, *, kappa_min: float = DEFAULT_KAPPA_MIN,
                              kappa_max: float = DEFAULT_KAPPA_MAX) -> np.ndarray:
    """Equation S17: kappa_p = kappa_min + q_p (kappa_max - kappa_min)."""
    if kappa_min <= 0:
        raise ValueError(f"kappa_min must be positive, got {kappa_min}")
    if kappa_max < kappa_min:
        raise ValueError(f"kappa_max ({kappa_max}) is below kappa_min ({kappa_min})")
    q = np.clip(np.asarray(quality, dtype=float), 0.0, 1.0)
    return kappa_min + q * (kappa_max - kappa_min)


def dirichlet_from_shares(share_mean: np.ndarray, share_sd: np.ndarray, *,
                          kappa_min: float = DEFAULT_KAPPA_MIN,
                          kappa_max: float = DEFAULT_KAPPA_MAX) -> float:
    """Concentration of the Dirichlet matching a set of share moments.

    For s ~ Dir(kappa * m), Var(s_j) = m_j (1 - m_j) / (kappa + 1), so each
    component implies kappa_j = m_j (1 - m_j) / var_j - 1. Averaging over the
    components is the usual moment-matching estimate; components with a
    degenerate share or zero spread carry no information and are skipped.
    """
    m = np.asarray(share_mean, dtype=float)
    sd = np.asarray(share_sd, dtype=float)
    ok = np.isfinite(m) & np.isfinite(sd) & (sd > 1e-12) & (m > 1e-9) & (m < 1 - 1e-9)
    if not ok.any():
        return float(kappa_min)
    est = m[ok] * (1.0 - m[ok]) / (sd[ok] ** 2) - 1.0
    est = est[np.isfinite(est) & (est > 0)]
    if not est.size:
        return float(kappa_min)
    return float(np.clip(np.mean(est), kappa_min, kappa_max))


def flows_to_node_allocation(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    n_flows: int,
    *,
    r_min: float = DEFAULT_R_MIN,
    r_max: float = DEFAULT_R_MAX,
    kappa_min: float = DEFAULT_KAPPA_MIN,
    kappa_max: float = DEFAULT_KAPPA_MAX,
    from_col: str = "from_node",
    flow_idx_col: str = "flow_idx",
) -> NodeAllocationData:
    """Map per-flow observations and pedigree onto node totals and shares.

    See the module docstring for the three cases (fully / partially /
    unobserved) and why partially observed nodes keep their per-flow terms.
    """
    layout = build_layout(df, n_flows, from_col=from_col, flow_idx_col=flow_idx_col)
    y, sg = combine_observations(obs, n_flows)
    observed = np.isfinite(y)

    # Relative sigma the likelihood actually applies, then S15 inverted.
    rel = np.full(n_flows, np.nan)
    np.divide(sg, np.maximum(np.abs(y), 1e-9), out=rel, where=observed)
    q_flow = np.full(n_flows, np.nan)
    q_flow[observed] = quality_index_from_rel_sigma(rel[observed], r_min=r_min, r_max=r_max)

    n_nodes = layout.n_nodes
    total_value = np.full(n_nodes, np.nan)
    total_sigma = np.full(n_nodes, np.nan)
    share_obs = np.full(n_flows, np.nan)
    quality = np.zeros(n_nodes)
    resid_idx: list = []
    rows = []

    for i, out in enumerate(layout.out_flows):
        node = int(layout.nodes[i])
        n_out = int(len(out))
        seen = observed[out]
        n_seen = int(seen.sum())

        # Node quality: unweighted mean over the scored outflows (author's choice).
        quality[i] = float(np.mean(q_flow[out][seen])) if n_seen else 0.0

        if n_seen == n_out and n_out > 0:
            tot = float(np.sum(y[out]))
            if tot > 1e-9:
                total_value[i] = tot
                total_sigma[i] = float(np.sqrt(np.sum(sg[out] ** 2)))
                if n_out > 1:
                    share_obs[out] = y[out] / tot
                status = "fully_observed"
            else:
                # A node whose observed outflows sum to zero has no usable split.
                status = "zero_total"
                resid_idx.extend(int(f) for f in out)
        elif n_seen:
            status = "partially_observed"
            resid_idx.extend(int(f) for f in out[seen])
        else:
            status = "unobserved"

        rows.append({"node": node, "n_out": n_out, "n_observed": n_seen,
                     "status": status, "quality_index": quality[i],
                     "total_value": total_value[i], "total_sigma": total_sigma[i]})

    kappa = pedigree_to_concentration(quality, kappa_min=kappa_min, kappa_max=kappa_max)
    meta = pd.DataFrame(rows)
    meta["kappa"] = kappa

    resid_idx_arr = np.array(sorted(set(resid_idx)), dtype=int)
    return NodeAllocationData(
        layout=layout,
        total_value=total_value, total_sigma=total_sigma,
        share_obs=share_obs, kappa=kappa, quality=quality,
        resid_flow_idx=resid_idx_arr,
        resid_value=y[resid_idx_arr] if resid_idx_arr.size else np.zeros(0),
        resid_sigma=sg[resid_idx_arr] if resid_idx_arr.size else np.zeros(0),
        meta=meta,
    )


def flows_to_totals_and_shares(x: np.ndarray, layout: NodeAllocationLayout,
                               *, floor: float = 1e-12) -> tuple:
    """Split a flow vector into (node totals, shares).

    The inverse of `layout.expand`. A node with zero throughput has no defined
    split; its shares are set uniform so the simplex constraint still holds.
    """
    x = np.asarray(x, dtype=float)
    totals = np.zeros(layout.n_nodes)
    shares = np.zeros(layout.n_flows)
    for i, out in enumerate(layout.out_flows):
        t = float(np.sum(np.abs(x[out])))
        totals[i] = t
        if t > floor:
            shares[out] = np.abs(x[out]) / t
        else:
            shares[out] = 1.0 / max(len(out), 1)
    return totals, shares


def expand_to_flows(totals: np.ndarray, shares: np.ndarray,
                    layout: NodeAllocationLayout) -> np.ndarray:
    """x_j = T_{from(j)} * s_j. Thin wrapper so callers need not hold the layout."""
    return layout.expand(totals, shares)
