"""
arcade_mfa_aluminum.pipeline_node_allocation
--------------------------------------------
Run the node-allocation formulation end to end.

This mirrors the stage order of the flow-based `pipeline.run` and reuses its
ingestion, mass-balance construction and transferred-prior artefacts, so the two
formulations see exactly the same data and differ only in parameterisation and
likelihood. It writes `posterior_mean.csv` in the same schema, so the existing
Sankey script works against a node-allocation run unchanged.

What it does NOT write is a `convergence_summary.csv`. The Laplace sampler
returns independent draws, so split-Rhat and ESS are vacuous here -- Rhat would
be 1.0 and ESS the draw count by construction, which would look like a clean
bill of health while measuring nothing. Sampler quality is instead bounded by
the analytic comparison in `tests/test_node_allocation.py` and reported through
the flat-direction warning.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.allocation import load_allocation_shares, node_totals_from_flows
from arcade_mfa_aluminum.graph import build_node_balance
from arcade_mfa_aluminum.inference.node_allocation_map import (
    NodeAllocationResult,
    NodeAllocationTerms,
    laplace_sample_node_allocation,
    solve_node_allocation_map,
)
from arcade_mfa_aluminum.node_allocation import (
    DEFAULT_KAPPA_MAX,
    DEFAULT_KAPPA_MIN,
    DEFAULT_R_MAX,
    DEFAULT_R_MIN,
    dirichlet_from_shares,
    flows_to_node_allocation,
)
from arcade_mfa_aluminum.paths import ensure_dir, long_path, prepare_output
from arcade_mfa_aluminum.pipeline import _ingest_2017, _ingest_2022, load_config


@dataclass
class NodeAllocationOutputs:
    run_dir: str
    result: NodeAllocationResult
    posterior_samples: np.ndarray
    posterior_mean: np.ndarray
    posterior_mean_csv_path: str
    node_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    conversion_meta: pd.DataFrame = field(default_factory=pd.DataFrame)


def _node_total_prior(prior_mean, prior_cov, layout):
    """Marginal prior on each node total from the per-flow transfer prior.

    T_p = sum_{j in O(p)} x_j, so the mean is the sum of means and the variance
    is the full covariance block summed -- the flows within a node are strongly
    correlated in the source posterior, so ignoring the off-diagonal terms would
    understate the prior spread badly.
    """
    mu = np.zeros(layout.n_nodes)
    sd = np.zeros(layout.n_nodes)
    for i, out in enumerate(layout.out_flows):
        mu[i] = float(np.sum(prior_mean[out]))
        var = float(prior_cov[np.ix_(out, out)].sum())
        sd[i] = float(np.sqrt(max(var, 1e-12)))
    return mu, sd


def _dirichlet_prior(shares_path, layout, kappa_min, kappa_max):
    """Per-node Dirichlet prior from the source-year allocation shares.

    Nodes absent from the source year, or whose outflow set has changed, get no
    prior -- the SI's uniform case -- rather than a mis-addressed one.
    """
    sh = load_allocation_shares(shares_path)
    by_flow_mean = {int(f): float(m) for f, m in zip(sh.flow_idx, sh.share_mean)}
    by_flow_sd = {int(f): float(s) for f, s in zip(sh.flow_idx, sh.share_sd)}
    mean = np.full(layout.n_flows, np.nan)
    kappa = np.full(layout.n_nodes, np.nan)
    for i, out in enumerate(layout.out_flows):
        if len(out) < 2:
            continue
        if not all(int(f) in by_flow_mean for f in out):
            continue
        m = np.array([by_flow_mean[int(f)] for f in out])
        s = np.array([by_flow_sd[int(f)] for f in out])
        tot = m.sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        m = m / tot
        mean[out] = m
        kappa[i] = dirichlet_from_shares(m, s, kappa_min=kappa_min, kappa_max=kappa_max)
    return mean, kappa


def run(config_path: str) -> NodeAllocationOutputs:
    cfg = load_config(config_path)
    run_cfg = cfg.get("run", {})
    name = run_cfg.get("name", "node_allocation")
    year = int(run_cfg.get("year", 2017))
    seed = int(run_cfg.get("seed", 42))
    na = cfg.get("node_allocation", {}) or {}

    # ---- 1. ingest, exactly as the flow pipeline does ---------------------
    if year == 2017:
        df, obs = _ingest_2017(cfg)
    elif year == 2022:
        df, obs = _ingest_2022(cfg)
    else:
        raise ValueError(f"unsupported run.year {year}; expected 2017 or 2022")
    n_flows = int(cfg.get("data", {}).get("n_flows", len(df)))
    if n_flows != len(df):
        raise ValueError(f"data.n_flows={n_flows} but the sheet has {len(df)} rows")

    # ---- 2. convert flow observations to node totals and shares -----------
    data = flows_to_node_allocation(
        df, obs, n_flows,
        r_min=float(na.get("r_min", DEFAULT_R_MIN)),
        r_max=float(na.get("r_max", DEFAULT_R_MAX)),
        kappa_min=float(na.get("kappa_min", DEFAULT_KAPPA_MIN)),
        kappa_max=float(na.get("kappa_max", DEFAULT_KAPPA_MAX)),
    )
    layout = data.layout
    counts = data.meta.status.value_counts().to_dict()
    print(f"[{name}] {layout.n_nodes} source nodes ({len(layout.branching)} branching), "
          f"{n_flows} flows")
    print(f"[{name}] conversion: {counts}")
    print(f"[{name}] node-total observations {data.n_total_obs}, Dirichlet share entries "
          f"{data.n_share_obs}, retained per-flow terms {len(data.resid_flow_idx)}")

    # ---- 3. warm start from the flow formulation --------------------------
    warm_path = na.get("warm_start_from")
    x0 = None
    if warm_path:
        p = long_path(warm_path)
        if os.path.exists(p):
            x0 = pd.read_csv(p)["posterior_mean"].to_numpy(dtype=float)
            if x0.size != n_flows:
                raise ValueError(
                    f"warm start {warm_path} has {x0.size} flows, expected {n_flows}")
            print(f"[{name}] warm start from {warm_path}")
        else:
            warnings.warn(f"[{name}] warm_start_from not found: {warm_path}", stacklevel=2)

    # ---- 4. soft mass balance --------------------------------------------
    mb = build_node_balance(df, n_flows)
    # Throughput reference for the tolerance. A warm start gives it directly;
    # otherwise use the observed node totals, falling back to 1 kt/y for nodes
    # with no observation. Using a vector of ones instead would set the
    # tolerance from the outflow COUNT and over-constrain every large node.
    if x0 is not None:
        totals_ref = node_totals_from_flows(x0, df)
    else:
        totals_ref = {int(layout.nodes[i]): (float(v) if np.isfinite(v) else 1.0)
                      for i, v in enumerate(data.total_value)}
    mb_cfg = cfg.get("mass_balance", {}) or {}
    mb_rel = float(mb_cfg.get("rel_sigma", 0.01))
    if mb_cfg.get("mode", "soft") != "soft":
        raise ValueError(
            "the node-allocation formulation supports mass_balance.mode: soft only; "
            "hard closure would need a nonlinear equality constraint, which is not "
            "implemented")
    mb_sigma = np.array([mb_rel * max(totals_ref.get(int(nd), 0.0), 1e-6) for nd in mb.nodes])
    print(f"[{name}] mass balance: SOFT, {mb.A.shape[0]} node rows at {100*mb_rel:.3g}% "
          f"of throughput")

    # ---- 5. transferred prior --------------------------------------------
    p_mu = p_sd = pr_share = pr_kappa = None
    prior_cfg = cfg.get("prior", {}) or {}
    if prior_cfg.get("enabled", False):
        pm = np.load(long_path(prior_cfg["mean_path"]))
        pc = np.load(long_path(prior_cfg["cov_path"]))
        if pm.size != n_flows:
            raise ValueError(f"prior has {pm.size} flows, expected {n_flows}; "
                             f"re-run the source-year config")
        strength = float(prior_cfg.get("strength", 1.0))
        if strength < 0:
            raise ValueError(f"prior.strength must be >= 0, got {strength}")
        p_mu, p_sd = _node_total_prior(pm, pc, layout)
        if strength == 0:
            p_mu = p_sd = None
            print(f"[{name}] prior.strength=0 -> transferred prior disabled")
        else:
            p_sd = p_sd / np.sqrt(strength)
            print(f"[{name}] node-total prior from {prior_cfg.get('source_year', '?')} "
                  f"(strength {strength})")
        shares_path = (cfg.get("allocation_constraints", {}) or {}).get("shares_path")
        if shares_path and strength > 0:
            pr_share, pr_kappa = _dirichlet_prior(
                shares_path, layout,
                float(na.get("kappa_min", DEFAULT_KAPPA_MIN)),
                float(na.get("kappa_max", DEFAULT_KAPPA_MAX)))
            n_pr = int(np.isfinite(pr_kappa).sum())
            print(f"[{name}] Dirichlet prior on {n_pr}/{len(layout.branching)} branching "
                  f"nodes; the rest are uniform")
            pr_kappa = np.where(np.isfinite(pr_kappa), pr_kappa, 0.0)

    terms = NodeAllocationTerms(
        layout=layout,
        total_value=data.total_value, total_sigma=data.total_sigma,
        share_obs=data.share_obs, kappa=data.kappa,
        resid_flow_idx=data.resid_flow_idx, resid_value=data.resid_value,
        resid_sigma=data.resid_sigma,
        mb_A=mb.A, mb_sigma=mb_sigma,
        prior_total_mean=p_mu, prior_total_sd=p_sd,
        prior_share_mean=pr_share, prior_kappa=pr_kappa,
    )

    # ---- 6. MAP -----------------------------------------------------------
    result = solve_node_allocation_map(
        terms, x0_flows=x0, maxiter=int(na.get("map_maxiter", 20000)))
    print(f"[{name}] MAP objective {result.objective:,.4f}, max |grad| "
          f"{result.grad_norm:.3g} -> {'converged' if result.converged else 'NOT CONVERGED'} "
          f"({result.n_iter} iterations)")
    if not result.converged:
        warnings.warn(
            f"[{name}] the MAP did not reach the stationarity threshold "
            f"(max |grad| {result.grad_norm:.3g}). Treat the point estimate as "
            f"provisional and raise node_allocation.map_maxiter.", stacklevel=2)
    resid = mb.A @ result.flows
    print(f"[{name}] mass-balance residual: max |kt/y| {np.abs(resid).max():,.3f}")

    # ---- 7. posterior ------------------------------------------------------
    samples = laplace_sample_node_allocation(
        result, n_draws=int(na.get("n_draws", 4000)), seed=seed,
        max_sd_z=float(na.get("max_sd_z", 10.0)))
    post_mean = samples.mean(axis=0)
    lo, med, hi = np.percentile(samples, [2.5, 50, 97.5], axis=0)
    print(f"[{name}] posterior: {samples.shape[0]} independent draws "
          f"(Laplace in the transformed space)")

    # ---- 8. outputs --------------------------------------------------------
    out_dir = (cfg.get("diagnostics", {}) or {}).get("output_dir", os.path.join("runs", name))
    ensure_dir(out_dir)

    table = df.copy()
    table["map_estimate"] = result.flows
    table["posterior_mean"] = post_mean
    table["posterior_median"] = med
    table["lower_bound"] = lo
    table["upper_bound"] = hi
    csv_path = (cfg.get("output", {}) or {}).get(
        "posterior_mean_csv", os.path.join(out_dir, "posterior_mean.csv"))
    table.to_csv(prepare_output(csv_path), index=False)

    npy_path = (cfg.get("output", {}) or {}).get(
        "posterior_samples_npy", os.path.join(out_dir, "posterior_samples.npy"))
    np.save(prepare_output(npy_path), samples)

    node_summary = data.meta.copy()
    node_summary["total_map"] = result.totals
    node_summary["kappa_likelihood"] = data.kappa
    if pr_kappa is not None:
        node_summary["kappa_prior"] = pr_kappa
    node_summary.to_csv(prepare_output(os.path.join(out_dir, "node_summary.csv")), index=False)

    shares_out = pd.DataFrame({
        "flow_idx": np.arange(n_flows),
        "from_node": df["from_node"].to_numpy(),
        "from_node_name": df["from_node_name"].to_numpy(),
        "to_node_name": df["to_node_name"].to_numpy(),
        "share_map": result.shares,
        "share_observed": data.share_obs,
    })
    shares_out.to_csv(prepare_output(os.path.join(out_dir, "allocation_shares_map.csv")),
                      index=False)

    print(f"[{name}] saved -> {csv_path}")
    return NodeAllocationOutputs(
        run_dir=out_dir, result=result, posterior_samples=samples,
        posterior_mean=post_mean, posterior_mean_csv_path=csv_path,
        node_summary=node_summary, conversion_meta=data.meta,
    )
