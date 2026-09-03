"""
arcade_mfa_aluminum.pipeline
----------------------------
End-to-end orchestration for one reconciliation run: load config -> ingest
the workbook -> build the mass-balance system -> solve the MAP problem ->
sample the constrained posterior -> (target years only) apply the
transferred prior and allocation ratios -> write diagnostics and outputs
to runs/<run_name>/.

Implements the flow-based formulation: the decision variables are the flow
magnitudes themselves, in linear (not log) space. See
docs/model_formulation.md.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from arcade_mfa_aluminum.adapters.aluminum.adapter import (
    assert_authoritative_2017,
    load_aluminum_2017_from_workbook,
    load_aluminum_2022_from_workbook,
)
from arcade_mfa_aluminum.allocation import (
    allocation_residual_report,
    build_allocation_constraints,
    compute_allocation_shares,
    load_allocation_shares,
    node_totals_from_flows,
    save_allocation_shares,
)
from arcade_mfa_aluminum.attribution import (
    classify_flows,
    observation_inventory,
    precision_shares,
    prior_posterior_comparison,
)
from arcade_mfa_aluminum.diagnostics import convergence_summary
from arcade_mfa_aluminum.graph import (
    build_node_balance,
    posterior_node_balance_diagnostics,
    soft_mass_balance_block,
)
from arcade_mfa_aluminum.inference.optimization import solve_map
from arcade_mfa_aluminum.inference.sampling import (
    laplace_sample_nullspace,
    truncated_gibbs_sample_nullspace,
)
from arcade_mfa_aluminum.paths import ensure_dir, long_path, prepare_output
from arcade_mfa_aluminum.priors.quality_sigma import sigma_for_observations
from arcade_mfa_aluminum.transfer import (
    build_2022_prior_from_2017,
    precision_from_cov,
    save_prior_npy,
)


@dataclass
class RunOutputs:
    run_dir: str
    x_map: np.ndarray
    posterior_mean: np.ndarray
    posterior_samples: np.ndarray
    node_balance_diagnostics: pd.DataFrame
    posterior_mean_csv_path: str
    convergence_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    aggregate_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    sampler_stats: dict = field(default_factory=dict)
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_config(config_path: str) -> dict:
    with open(long_path(config_path), "r") as f:
        return yaml.safe_load(f)


def _build_bounds(y_obs: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    b = cfg.get("bounds", {})
    lo_frac = b.get("observed_lower_frac", 0.2)
    hi_frac = b.get("observed_upper_frac", 1.8)
    obs_mask = np.isfinite(y_obs)
    max_obs = np.nanmax(y_obs) if obs_mask.any() else 1.0
    sup_bound = max(1.0, max_obs * hi_frac)

    x_lb = np.where(obs_mask, lo_frac * y_obs, 0.0)
    x_ub = np.where(obs_mask, hi_frac * y_obs, sup_bound)

    # A flow reported as exactly zero would give lb = ub = 0, an empty interior.
    # Widen the UPPER bound rather than lowering the lower one, so that flows
    # stay non-negative.
    #
    # The width matters. A reported zero usually means "not measured" or "below
    # reporting threshold" rather than "provably absent", so pinning such a flow
    # to a numerically negligible interval asserts far more certainty than the
    # data support, and prevents mass balance from routing anything through it.
    # `bounds.zero_flow_upper` sets how much room a reported zero is given, in
    # mass units. Set it to 0 to recover an effectively hard zero.
    zero_upper = float(b.get("zero_flow_upper", 1.0))
    degenerate = (x_ub - x_lb) < max(zero_upper, 1e-6)
    x_ub = np.where(degenerate, x_lb + max(zero_upper, 1e-6), x_ub)
    return x_lb, x_ub


def _ingest_2017(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 2017 flow table and every observation reported against it.

    Returns (canonical flow dataframe, long-form observation table with a
    `sigma` column).
    """
    data_cfg = cfg["data"]
    workbook_path = data_cfg.get("workbook_path", "data/raw/aluminum/US aluminum flows.xlsx")
    sheet_name = data_cfg.get("source_name", "2017 data")
    assert_authoritative_2017(sheet_name)
    table = load_aluminum_2017_from_workbook(workbook_path, sheet_name=sheet_name)
    return table.df, _observation_sigma(table.observations, cfg)


def _ingest_2022(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 2022 flow table and every observation reported against it."""
    data_cfg = cfg["data"]
    workbook_path = data_cfg.get("workbook_path", "data/raw/aluminum/US aluminum flows.xlsx")
    sheet_name = data_cfg.get("source_name", "2022 results")
    table = load_aluminum_2022_from_workbook(workbook_path, sheet_name=sheet_name)
    return table.df, _observation_sigma(table.observations, cfg)


def _observation_sigma(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach a `sigma` column to the long-form observation table.

    Every reported value is retained, including several on the same flow; the
    likelihood reconciles them by relative precision rather than the loader
    choosing between them.
    """
    obs_cfg = cfg.get("observation_sigma", {})
    src_cfg = cfg.get("observation_sources", {}) or {}
    multipliers = {
        name: float(spec.get("sigma_multiplier", 1.0))
        for name, spec in src_cfg.items()
        if isinstance(spec, dict)
    }
    obs = obs.copy()

    # A source can be held out entirely (`observation_sources.<name>.enabled:
    # false`), which is how leave-one-source-out validation is run. The rows are
    # dropped rather than given a large sigma: a large sigma still pulls the
    # posterior a little, so the held-out values would not be truly unseen and
    # the validation would flatter itself.
    disabled = sorted(
        name for name, spec in src_cfg.items()
        if isinstance(spec, dict) and not spec.get("enabled", True)
    )
    if disabled:
        keep = ~obs["source"].astype(str).isin(disabled)
        if not keep.any():
            raise ValueError(
                f"observation_sources: holding out {disabled} leaves no "
                f"observations at all."
            )
        print(f"[holdout] per-flow sources withheld: {', '.join(disabled)} "
              f"({int((~keep).sum())} of {len(obs)} observations)")
        obs = obs[keep].reset_index(drop=True)
    obs["sigma"] = sigma_for_observations(
        obs,
        mapping=obs_cfg.get("pedigree_mapping", {}),
        fallback_rel_sigma=obs_cfg.get("observed_rel_sigma", 0.10),
        source_multipliers=multipliers,
        # Same allowance the bounds give a reported zero, so the likelihood and
        # the box agree about how much room such a flow has.
        zero_value_scale=float(cfg.get("bounds", {}).get("zero_flow_upper", 1.0)),
    )

    # Per-flow uncertainty overrides. A pedigree score describes a source in
    # general; occasionally a specific quantity is known to be better or worse
    # measured than its source-level score implies (customs-tracked trade being
    # the usual example). Each override records the flow, an optional source
    # filter, and a justification so the choice is auditable rather than buried.
    for ov in cfg.get("observation_overrides", []) or []:
        m = obs["flow_idx"] == int(ov["flow_idx"])
        if "source" in ov:
            m &= obs["source"].astype(str) == str(ov["source"])
        if not m.any():
            warnings.warn(
                f"observation_overrides: no observation matches flow_idx="
                f"{ov.get('flow_idx')} source={ov.get('source', 'any')}; "
                f"the override had no effect.",
                stacklevel=2,
            )
            continue
        if "relative_sigma" in ov:
            # Mirror the zero handling in `sigma_for_observations`. A purely
            # relative sigma collapses at zero, so applying one to a reported
            # zero would pin the flow to a hard zero -- far stronger than any
            # override is meant to assert, and stronger than the pedigree path
            # would ever produce for the same value.
            rel_ov = float(ov["relative_sigma"])
            v_ov = obs.loc[m, "value"].abs().to_numpy(dtype=float)
            z_scale = float(cfg.get("bounds", {}).get("zero_flow_upper", 1.0))
            obs.loc[m, "sigma"] = np.maximum(
                np.where(v_ov > 1e-9, rel_ov * v_ov, rel_ov * z_scale), 1e-3
            )
        elif "sigma_multiplier" in ov:
            obs.loc[m, "sigma"] = obs.loc[m, "sigma"] * float(ov["sigma_multiplier"])
        else:
            raise ValueError(
                f"observation_overrides entry for flow_idx={ov.get('flow_idx')} "
                f"needs either relative_sigma or sigma_multiplier."
            )
    return obs


def _representative_values(obs: pd.DataFrame, n_flows: int) -> np.ndarray:
    """Inverse-variance weighted mean observation per flow, NaN where none.

    Used only to set the box bounds, which need one representative magnitude
    per flow. Weighting matches the likelihood, so the bounds sit around the
    same value the observations pull toward.
    """
    num = np.zeros(n_flows)
    den = np.zeros(n_flows)
    if len(obs):
        w = 1.0 / np.maximum(obs["sigma"].to_numpy(dtype=float) ** 2, 1e-12)
        idx = obs["flow_idx"].to_numpy(dtype=int)
        np.add.at(num, idx, w * obs["value"].to_numpy(dtype=float))
        np.add.at(den, idx, w)
    out = np.full(n_flows, np.nan)
    seen = den > 0
    out[seen] = num[seen] / den[seen]
    return out


def run(config_path: str) -> RunOutputs:
    cfg = load_config(config_path)
    run_name = cfg["run"]["name"]
    year = cfg["run"]["year"]
    seed = cfg["run"].get("seed", 42)

    space = cfg.get("formulation", {}).get("space", "linear")
    if space == "log":
        raise NotImplementedError(
            "formulation.space='log' is not implemented. The objective in "
            "inference.optimization is quadratic in linear flow space. "
            "Use space='linear'."
        )
    if space != "linear":
        raise ValueError(f"Unknown formulation.space={space!r}. Supported: 'linear'.")

    if year == 2017:
        df, obs = _ingest_2017(cfg)
    elif year == 2022:
        df, obs = _ingest_2022(cfg)
    else:
        raise ValueError(f"Unsupported run year: {year}")

    n_flows = cfg["data"].get("n_flows", len(df))
    # Bounds need one magnitude per flow; the likelihood uses every observation.
    y_obs = _representative_values(obs, n_flows)
    sigma_obs = np.full(n_flows, np.inf)
    x_lb, x_ub = _build_bounds(y_obs, cfg)

    by_flow = obs.groupby("flow_idx").size() if len(obs) else pd.Series(dtype=int)
    n_multi = int((by_flow > 1).sum())
    print(f"[{run_name}] {len(obs)} observations from {obs['source'].nunique()} source(s) "
          f"over {len(by_flow)}/{n_flows} flows; {n_multi} flows carry more than one.")
    map_obs = dict(
        obs_idx=obs["flow_idx"].to_numpy(dtype=int),
        obs_value=obs["value"].to_numpy(dtype=float),
        obs_sigma=obs["sigma"].to_numpy(dtype=float),
    )

    mb = build_node_balance(df, n_flows)
    print(f"[{run_name}] {n_flows} flows, {mb.A.shape[0]} mass-balance constraints "
          f"({len(mb.nodes)} internal nodes).")

    # Guard against an unnoticed topology change in the source workbook.
    expected_nodes = cfg["data"].get("n_internal_nodes")
    if expected_nodes is not None and len(mb.nodes) != expected_nodes:
        raise ValueError(
            f"data.n_internal_nodes={expected_nodes} but the flow table yields "
            f"{len(mb.nodes)} internal nodes. The source topology has changed; "
            f"confirm the workbook sheet before trusting downstream results."
        )

    prior_mean, inv_prior_cov, prior_cov = None, None, None
    prior_cfg = cfg.get("prior", {})
    if prior_cfg.get("enabled", False):
        prior_mean = np.load(long_path(prior_cfg["mean_path"]))
        prior_cov = np.load(long_path(prior_cfg["cov_path"]))
        # Guard against a prior built for a different flow set: flow_idx is
        # renumbered whenever nodes are added or removed, so a stale artifact
        # would misalign every flow without raising.
        if prior_mean.shape[0] != n_flows or prior_cov.shape[0] != n_flows:
            raise ValueError(
                f"Prior was built for {prior_mean.shape[0]} flows but this run has "
                f"{n_flows}. The flow set changed (e.g. nodes were removed and "
                f"flow_idx renumbered). Re-run the {prior_cfg.get('source_year', 2017)} "
                f"config to regenerate {prior_cfg['mean_path']} and "
                f"{prior_cfg['cov_path']} before using them."
            )
        # Not np.linalg.inv: the transferred covariance is singular in the
        # mass-balance directions. See transfer.precision_from_cov.
        inv_prior_cov = precision_from_cov(prior_cov)
        # `prior.strength` scales how strongly the source-year posterior binds
        # this year: 1.0 uses it as derived, values below 1 weaken it, and 0
        # removes it entirely. Sweeping this separates results that are driven
        # by this year's observations from those carried by the prior.
        strength = float(prior_cfg.get("strength", 1.0))
        if strength < 0:
            raise ValueError(f"prior.strength must be >= 0, got {strength}")
        if strength != 1.0:
            inv_prior_cov = inv_prior_cov * strength
            print(f"[{run_name}] prior strength scaled by {strength} "
                  f"({'disabled' if strength == 0 else 'weakened' if strength < 1 else 'strengthened'}).")
        prior_rank = int(np.linalg.matrix_rank(inv_prior_cov))
        print(f"[{run_name}] loaded prior from {prior_cfg['source_year']} "
              f"(rank {prior_rank}/{prior_cov.shape[0]}; the remaining directions "
              f"are the mass-balance subspace, left to the equality constraints).")
        if "inflation_factor" in prior_cfg:
            warnings.warn(
                f"[{run_name}] prior.inflation_factor={prior_cfg['inflation_factor']} in the "
                f"config has NO effect: covariance inflation is applied when the prior is "
                f"built in the source-year run (output.transfer_inflation_factor). Re-run the "
                f"{prior_cfg.get('source_year', 2017)} config to change it.",
                stacklevel=2,
            )

    map_cfg = cfg.get("inference", {}).get("map", {})
    map_kwargs = dict(
        **map_obs,
        prior_mean=prior_mean, inv_prior_cov=inv_prior_cov,
        maxiter=map_cfg.get("maxiter", 20000),
        method=map_cfg.get("method", "trust-constr"),
    )

    # ---- soft constraint blocks --------------------------------------------
    # Two relaxed blocks stack into one penalty term: mass balance (optional,
    # see mass_balance.mode) and transferred allocation ratios. Both are
    # linearized about node totals taken from a MAP warm-up, so a single
    # warm-up serves both. Box bounds always stay hard.
    mb_cfg = cfg.get("mass_balance", {})
    mb_mode = mb_cfg.get("mode", "soft")
    if mb_mode not in ("soft", "hard"):
        raise ValueError(
            f"Unknown mass_balance.mode={mb_mode!r}. Supported: 'soft', 'hard'."
        )
    alloc_cfg = cfg.get("allocation_constraints", {})
    alloc_enabled = bool(alloc_cfg.get("enabled", False))

    # In soft mode the equality rows move into the penalty, so the solver and
    # the sampler must NOT also see them as hard constraints.
    hard_A = mb.A if mb_mode == "hard" else np.zeros((0, n_flows))
    hard_b = mb.b if mb_mode == "hard" else np.zeros(0)

    ref_x = None
    if mb_mode == "soft" or alloc_enabled:
        linearization = alloc_cfg.get("linearization", "map_warmup")
        if linearization == "map_warmup":
            # The warm-up only supplies node-total SCALES for linearizing the
            # soft blocks, so it does not need to converge; a loose iteration
            # cap halves total runtime with no effect on the final solution.
            warm_kwargs = dict(map_kwargs)
            warm_kwargs["maxiter"] = map_cfg.get("warmup_maxiter", 300)
            print(f"[{run_name}] warm-up MAP ({warm_kwargs['maxiter']} iters) "
                  f"to set node-total reference scales...")
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                warm = solve_map(y_obs, sigma_obs, mb.A, mb.b, x_lb, x_ub, **warm_kwargs)
            ref_x = warm.x_map
        elif linearization == "prior_mean":
            if prior_mean is None:
                raise ValueError(
                    "linearization='prior_mean' requires prior.enabled: true, "
                    "but no prior was loaded."
                )
            ref_x = prior_mean
        elif linearization == "observed":
            ref_x = np.nan_to_num(y_obs, nan=0.0)
        else:
            raise ValueError(
                f"Unknown allocation_constraints.linearization={linearization!r}. "
                f"Supported: 'map_warmup', 'prior_mean', 'observed'."
            )
        totals = node_totals_from_flows(ref_x, df)

    soft_blocks = []          # (label, A, b, sigma)

    if mb_mode == "soft":
        mb_rel = mb_cfg.get("rel_sigma", 0.01)
        A_mb, b_mb, s_mb = soft_mass_balance_block(mb, totals, rel_sigma=mb_rel)
        if A_mb.shape[0] == 0:
            raise ValueError(
                "mass_balance.mode='soft' but no node carries enough throughput "
                "to define a tolerance; check the warm-up solution."
            )
        soft_blocks.append(("mass_balance", A_mb, b_mb, s_mb))
        print(f"[{run_name}] mass balance: SOFT, {A_mb.shape[0]}/{mb.A.shape[0]} node rows "
              f"penalized at {mb_rel:.1%} of node throughput "
              f"(sigma {s_mb.min():.3g}-{s_mb.max():.3g}).")
    else:
        print(f"[{run_name}] mass balance: HARD equality on {mb.A.shape[0]} node rows.")

    alloc_con = None
    if alloc_enabled:
        shares = load_allocation_shares(alloc_cfg["shares_path"])
        alloc_con = build_allocation_constraints(
            df, shares, totals, n_flows=n_flows,
            relaxation=alloc_cfg.get("relaxation", 2.0),
            min_share_sigma=alloc_cfg.get("min_share_sigma", 0.02),
            max_share_sigma=alloc_cfg.get("max_share_sigma", 0.5),
            exclude_nodes=tuple(alloc_cfg.get("exclude_nodes", []) or ()),
        )
        if alloc_con.n_rows == 0:
            warnings.warn(
                f"[{run_name}] allocation_constraints.enabled is true but no usable "
                f"constraints were built (no shared branching nodes, or all excluded).",
                stacklevel=2,
            )
        else:
            soft_blocks.append(
                ("allocation", alloc_con.R, np.zeros(alloc_con.n_rows), alloc_con.tau)
            )
            print(f"[{run_name}] allocation: {alloc_con.n_rows} soft rows over "
                  f"{alloc_con.n_nodes} branching nodes "
                  f"(relaxation={alloc_cfg.get('relaxation', 2.0)}, "
                  f"share sigma {alloc_con.meta['share_sigma'].min():.3f}-"
                  f"{alloc_con.meta['share_sigma'].max():.3f}).")

    # ---- aggregate observations ---------------------------------------------
    # Some sources report a total spanning several model flows. Constraining the
    # sum states exactly what was published; splitting it across flows in
    # proportion to another source would invent detail and partly restate the
    # source being compared against.
    agg_specs = list(cfg.get("aggregate_observations", []) or [])
    for extra in (cfg.get("aggregate_observation_files", []) or []):
        with open(long_path(extra)) as f:
            loaded = yaml.safe_load(f) or {}
        agg_specs.extend(loaded.get("aggregate_observations", []) or [])

    # Aggregates honour the same hold-out switch as per-flow observations, so a
    # source reporting only totals (the Aluminum Association) can be withheld
    # and scored the same way.
    _src_cfg = cfg.get("observation_sources", {}) or {}
    _disabled = {
        name for name, spec in _src_cfg.items()
        if isinstance(spec, dict) and not spec.get("enabled", True)
    }
    if _disabled and agg_specs:
        before = len(agg_specs)
        agg_specs = [a for a in agg_specs if str(a.get("source", "")) not in _disabled]
        if before != len(agg_specs):
            print(f"[holdout] aggregate totals withheld: "
                  f"{before - len(agg_specs)} of {before}")

    if agg_specs:
        rows, targets, sigmas, meta = [], [], [], []
        for spec in agg_specs:
            idx = np.asarray(spec["flows"], dtype=int)
            if idx.size == 0:
                raise ValueError(f"aggregate observation {spec.get('name')!r} lists no flows")
            if idx.min() < 0 or idx.max() >= n_flows:
                raise ValueError(
                    f"aggregate observation {spec.get('name')!r} references flow_idx "
                    f"outside [0, {n_flows - 1}]: [{idx.min()}, {idx.max()}]. The flow "
                    f"set may have changed since the observation was recorded."
                )
            row = np.zeros(n_flows)
            row[idx] = 1.0
            value = float(spec["value"])
            if "sigma" in spec:
                sg = float(spec["sigma"])
            else:
                sg = float(spec.get("relative_sigma", 0.10)) * max(abs(value), 1e-6)
            rows.append(row)
            targets.append(value)
            sigmas.append(max(sg, 1e-6))
            meta.append({"name": spec.get("name", ""), "source": spec.get("source", ""),
                         "n_flows": int(idx.size), "value": value, "sigma": sg,
                         "justification": spec.get("justification", "")})
        agg_A = np.vstack(rows)
        soft_blocks.append(("aggregate_observations", agg_A,
                            np.asarray(targets, dtype=float),
                            np.asarray(sigmas, dtype=float)))
        agg_meta = pd.DataFrame(meta)
        print(f"[{run_name}] aggregate observations: {len(agg_specs)} totals from "
              f"{agg_meta['source'].nunique()} source(s) spanning "
              f"{int(agg_meta['n_flows'].sum())} flow slots.")
    else:
        agg_meta = pd.DataFrame()

    soft_precisions = {
        label: (A.T * (1.0 / np.maximum(sg ** 2, 1e-12))) @ A
        for label, A, _b, sg in soft_blocks
    }

    if soft_blocks:
        map_kwargs.update(
            soft_A=np.vstack([b[1] for b in soft_blocks]),
            soft_b=np.concatenate([b[2] for b in soft_blocks]),
            soft_sigma=np.concatenate([b[3] for b in soft_blocks]),
        )

    map_result = solve_map(y_obs, sigma_obs, hard_A, hard_b, x_lb, x_ub, **map_kwargs)
    if not map_result.success:
        warnings.warn(f"[{run_name}] MAP solver did not fully converge: {map_result.message}")

    # Report the residual against the FULL balance system regardless of mode:
    # in soft mode this is a genuine diagnostic rather than machine epsilon.
    resid = mb.A @ map_result.x_map if mb.A.shape[0] > 0 else np.array([])
    if resid.size > 0:
        tp = np.array([max(totals.get(int(nd), 0.0), 1e-12) for nd in mb.nodes]) \
            if ref_x is not None else np.ones(len(mb.nodes))
        print(f"[{run_name}] MAP mass-balance residual: "
              f"max |kt/y| {np.abs(resid).max():.4g}, "
              f"max |% of node throughput| {np.abs(resid / tp).max() * 100:.4g}%")

    sampling_cfg = cfg.get("inference", {}).get("sampling", {})
    # An unsupported value raises rather than silently falling back.
    method = cfg.get("inference", {}).get("method", "truncated_gibbs")
    n_chains = sampling_cfg.get("n_chains", 4)
    n_draws = sampling_cfg.get("n_draws", 2000)

    if method == "truncated_gibbs":
        posterior = truncated_gibbs_sample_nullspace(
            map_result.x_map, map_result.Q, hard_A, x_lb, x_ub,
            n_chains=n_chains, n_draws=n_draws,
            n_tune=sampling_cfg.get("n_tune", 500),
            thin=sampling_cfg.get("thin", 1),
            seed=seed,
        )
    elif method == "laplace_nullspace":
        warnings.warn(
            f"[{run_name}] inference.method='laplace_nullspace' clips draws to the "
            f"bounds and then projects onto the mass-balance subspace, so the draws "
            f"it returns are not guaranteed to satisfy the bounds. Retained only to "
            f"reproduce results from the original notebooks; use 'truncated_gibbs' "
            f"otherwise.",
            stacklevel=2,
        )
        posterior = laplace_sample_nullspace(
            map_result.x_map, map_result.Q, hard_A, x_lb, x_ub,
            n_chains=n_chains, n_draws=n_draws, seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown inference.method={method!r}. "
            f"Supported: 'truncated_gibbs', 'laplace_nullspace'."
        )

    # Draws must satisfy both constraint sets; verify rather than assume.
    viol_lo = (posterior.samples < x_lb - 1e-6).sum()
    viol_hi = (posterior.samples > x_ub + 1e-6).sum()
    if viol_lo or viol_hi:
        warnings.warn(
            f"[{run_name}] {viol_lo + viol_hi} of {posterior.samples.size} posterior "
            f"draws fall outside the box bounds ({viol_lo} low, {viol_hi} high).",
            stacklevel=2,
        )
    print(f"[{run_name}] sampler={posterior.method}, {n_chains} chains x {n_draws} draws, "
          f"bound violations: {viol_lo + viol_hi}/{posterior.samples.size}")
    if posterior.stats:
        st = posterior.stats
        print(f"[{run_name}] sampler moves: {st['n_dimensions']}-dim, "
              f"coordinate accept {st['coordinate_move_rate']:.1%}, "
              f"hit-and-run accept {st['hit_and_run_move_rate']:.1%}, "
              f"degenerate skips {st['skipped_degenerate_updates']:,}")
    flat_samples = posterior.samples.reshape(-1, posterior.samples.shape[-1])
    posterior_mean = flat_samples.mean(axis=0)

    diag_cfg = cfg.get("diagnostics", {})
    out_dir = diag_cfg.get("output_dir", f"runs/{run_name}")
    ensure_dir(out_dir)

    # Split-Rhat and ESS distinguish a converged chain from one that merely ran.
    # Written on every run so that unreliable intervals are visible.
    conv = pd.DataFrame()
    if diag_cfg.get("compute_convergence_summary", True):
        conv = convergence_summary(posterior.samples, labels=df["flow_idx"].to_numpy())
        conv.insert(1, "flow_label",
                    df["from_node_name"].astype(str) + " -> " + df["to_node_name"].astype(str))
        conv_path = os.path.join(out_dir, "convergence_summary.csv")
        conv.to_csv(prepare_output(conv_path), index=False)
        n_bad = int((~conv["converged"]).sum())
        print(f"[{run_name}] convergence: Rhat median {conv['rhat'].median():.4f} / "
              f"max {conv['rhat'].max():.4f}, ESS min {conv['ess'].min():.0f} / "
              f"median {conv['ess'].median():.0f}; {n_bad}/{len(conv)} flows flagged.")
        if n_bad:
            worst = conv.nlargest(min(5, n_bad), "rhat")[["flow_idx", "rhat", "ess"]]
            warnings.warn(
                f"[{run_name}] {n_bad} flows failed the convergence check "
                f"(Rhat > 1.05 or ESS < 400). Their credible intervals are not "
                f"reliable. Worst: " + worst.to_string(index=False),
                stacklevel=2,
            )

    # Did the target year actually honour the transferred ratios, or fight them?
    # A large z_vs_tolerance means this year's own data disagrees with the
    # source-year split -- a finding to report, not a defect to hide.
    alloc_resid = pd.DataFrame()
    if alloc_con is not None and alloc_con.n_rows > 0:
        alloc_resid = allocation_residual_report(flat_samples, alloc_con)
        alloc_path = os.path.join(out_dir, "allocation_residuals.csv")
        alloc_resid.to_csv(prepare_output(alloc_path), index=False)
        z = alloc_resid["z_vs_tolerance"].abs()
        print(f"[{run_name}] allocation residuals: median |dev| "
              f"{alloc_resid['share_dev_abs_mean'].median():.4f} share units, "
              f"max |z| {z.max():.2f}; {int((z > 2).sum())}/{len(z)} rows beyond 2 sigma.")
        if (z > 3).any():
            worst = (alloc_resid.reindex(z.sort_values(ascending=False).index)
                     .head(5)[["node", "flow_idx", "target_share",
                               "share_dev_mean", "z_vs_tolerance"]])
            warnings.warn(
                f"[{run_name}] {int((z > 3).sum())} allocation rows deviate beyond "
                f"3 sigma: this year's data disagrees with the transferred split. "
                f"Worst: " + worst.to_string(index=False),
                stacklevel=2,
            )

    # What actually constrains each flow: observations, the transferred prior,
    # or the structural (mass-balance / allocation) blocks. R1-5 and R2-9 both
    # ask for results to be separated on exactly this basis.
    attribution = pd.DataFrame()
    if diag_cfg.get("compute_attribution", True):
        shares = precision_shares(map_result.q_obs_diag, map_result.Q_prior, soft_precisions)
        inv = observation_inventory(obs, n_flows)
        attribution = classify_flows(shares, inv)
        cmp_df = prior_posterior_comparison(flat_samples, prior_mean, prior_cov)
        attribution = attribution.merge(cmp_df, on="flow_idx", how="left")
        attribution.insert(1, "flow_label",
                           df["from_node_name"].astype(str) + " -> " + df["to_node_name"].astype(str))
        attribution.to_csv(prepare_output(os.path.join(out_dir, "attribution.csv")), index=False)
        counts = attribution["determined_by"].value_counts().to_dict()
        print(f"[{run_name}] flow attribution: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    # Every observation that entered the likelihood, with its source, pedigree
    # scores and derived sigma. R2-9 asks for exactly this: readers should be
    # able to see what was fed in and how much each input was trusted.
    if diag_cfg.get("write_source_inventory", True):
        inv_out = obs.copy()
        inv_out.insert(1, "flow_label",
                       df["from_node_name"].astype(str).to_numpy()[inv_out["flow_idx"].to_numpy()]
                       + " -> "
                       + df["to_node_name"].astype(str).to_numpy()[inv_out["flow_idx"].to_numpy()])
        inv_out["relative_sigma"] = (
            inv_out["sigma"] / inv_out["value"].abs().clip(lower=1e-9)
        )
        inv_out.to_csv(prepare_output(os.path.join(out_dir, "source_inventory.csv")), index=False)

    # How each reported total compares with the reconciled sum. Where two
    # sources disagree this is where the resolution becomes visible.
    if len(agg_meta):
        recon = []
        for spec, (_, row) in zip(agg_specs, agg_meta.iterrows()):
            idx = np.asarray(spec["flows"], dtype=int)
            draws = flat_samples[:, idx].sum(axis=1)
            lo, med, hi = np.percentile(draws, [2.5, 50, 97.5])
            recon.append({**row.to_dict(), "reconciled_median": med,
                          "reconciled_ci_lo": lo, "reconciled_ci_hi": hi,
                          "deviation": med - row["value"],
                          "z_vs_sigma": (med - row["value"]) / max(row["sigma"], 1e-9)})
        agg_meta = pd.DataFrame(recon)
        agg_meta.to_csv(
            prepare_output(os.path.join(out_dir, "aggregate_observations.csv")), index=False)
        worst = agg_meta.reindex(agg_meta.z_vs_sigma.abs().sort_values(ascending=False).index)
        print(f"[{run_name}] aggregate observations vs reconciled: max |z| "
              f"{agg_meta.z_vs_sigma.abs().max():.2f} "
              f"({worst.iloc[0]['name']}: reported {worst.iloc[0]['value']:,.0f}, "
              f"reconciled {worst.iloc[0]['reconciled_median']:,.0f})")

    node_diag = pd.DataFrame()
    if diag_cfg.get("compute_node_balance", True):
        node_diag = posterior_node_balance_diagnostics(flat_samples, df)
        node_diag_path = os.path.join(out_dir, "node_balance_diagnostics.csv")
        node_diag.to_csv(prepare_output(node_diag_path), index=False)

    out_df = df.copy()
    out_df["map_estimate"] = map_result.x_map
    out_df["posterior_mean"] = posterior_mean
    out_df["lower_bound"] = x_lb
    out_df["upper_bound"] = x_ub
    posterior_mean_csv_path = cfg.get("output", {}).get(
        "posterior_mean_csv", os.path.join(out_dir, "posterior_mean.csv")
    )
    out_df.to_csv(prepare_output(posterior_mean_csv_path), index=False)
    print(f"[{run_name}] saved posterior mean table -> {posterior_mean_csv_path}")

    samples_path = cfg.get("output", {}).get(
        "posterior_samples_npy", os.path.join(out_dir, "posterior_samples.npy")
    )
    np.save(prepare_output(samples_path), flat_samples)

    if year == 2017 and cfg.get("output", {}).get("save_prior_for_transfer", False):
        transfer_prior = build_2022_prior_from_2017(
            posterior.samples,
            inflation_factor=cfg.get("output", {}).get("transfer_inflation_factor", 1.5),
        )
        mean_out = cfg["output"]["prior_mean_out"]
        cov_out = cfg["output"]["prior_cov_out"]
        save_prior_npy(transfer_prior, mean_out, cov_out)
        print(f"[{run_name}] saved 2022 transfer prior -> {mean_out}, {cov_out}")

    # Export this year's node allocation ratios for transfer to a later vintage.
    if cfg.get("output", {}).get("save_allocation_shares", False):
        shares_out = cfg["output"].get(
            "allocation_shares_out",
            f"data/canonical/aluminum/allocation_shares_{year}.npz",
        )
        shares, _w = compute_allocation_shares(
            flat_samples, df,
            min_outflows=cfg["output"].get("allocation_min_outflows", 2),
        )
        save_allocation_shares(shares, shares_out)
        shares_csv = os.path.join(out_dir, "allocation_shares.csv")
        shares.to_frame().to_csv(prepare_output(shares_csv), index=False)
        n_nodes = len(np.unique(shares.node)) if len(shares) else 0
        print(f"[{run_name}] saved allocation shares -> {shares_out} "
              f"({len(shares)} flows across {n_nodes} branching nodes; "
              f"median share sd {np.median(shares.share_sd):.4f})")

    return RunOutputs(
        run_dir=out_dir,
        x_map=map_result.x_map,
        posterior_mean=posterior_mean,
        posterior_samples=flat_samples,
        node_balance_diagnostics=node_diag,
        posterior_mean_csv_path=posterior_mean_csv_path,
        convergence_summary=conv,
        allocation_residuals=alloc_resid,
        attribution=attribution,
        aggregate_observations=agg_meta,
        sampler_stats=posterior.stats or {},
        observations=obs,
    )
