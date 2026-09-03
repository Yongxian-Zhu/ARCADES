"""
scripts/build_si_tables.py
--------------------------
Assemble the Supporting Information tables from completed run directories.

Four tables are produced, each answering a specific transparency question:

`reconciliation_table.csv`
    One row per flow: every raw observation with its source, the prior estimate,
    the posterior median and 95% credible interval, and a label saying what
    determined the value. This is the table that shows how conflicting sources
    were resolved, and which quantities the observations actually constrained
    rather than the prior or the network structure.

`diagnostics_table.csv`
    Convergence and sampler behaviour per flow -- split-Rhat, effective sample
    size, and the interval -- together with the run-level sampler move rates and
    mass-balance residual summary.

`node_balance_table.csv`
    Per node: the posterior residual in kt/y and as a percentage of node
    throughput, with credible intervals. Shows where the system balances tightly
    and where reported data genuinely disagree.

`source_inventory.csv`
    Every observation that entered the likelihood, with source, pedigree scores
    and the derived sigma.

Usage:
    python scripts/build_si_tables.py --runs runs/aluminum_2017_baseline runs/aluminum_2022_update
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.paths import long_path, prepare_output


def _read(run_dir: str, name: str) -> pd.DataFrame | None:
    path = long_path(os.path.join(run_dir, name))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _label(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


def build_reconciliation_table(run_dir: str, ci_level: float = 0.95) -> pd.DataFrame:
    """Raw observations against prior and posterior, per flow."""
    mean = _read(run_dir, "posterior_mean.csv")
    if mean is None:
        raise FileNotFoundError(f"{run_dir}: posterior_mean.csv not found")
    samples = np.load(long_path(os.path.join(run_dir, "posterior_samples.npy")))
    attribution = _read(run_dir, "attribution.csv")
    inventory = _read(run_dir, "source_inventory.csv")

    alpha = 1.0 - ci_level
    lo, med, hi = np.percentile(samples, [alpha / 2 * 100, 50, (1 - alpha / 2) * 100], axis=0)

    out = pd.DataFrame({
        "flow_idx": mean["flow_idx"],
        "from_node": mean["from_node_name"],
        "to_node": mean["to_node_name"],
        "posterior_median": med,
        "posterior_ci_lo": lo,
        "posterior_ci_hi": hi,
        "posterior_ci_width": hi - lo,
    })
    out["relative_ci_width"] = out["posterior_ci_width"] / out["posterior_median"].abs().clip(lower=1e-9)

    # Observations, one column per source, so disagreements are visible.
    if inventory is not None and len(inventory):
        wide = inventory.pivot_table(index="flow_idx", columns="source",
                                     values="value", aggfunc="first")
        wide.columns = [f"obs: {c}" for c in wide.columns]
        out = out.merge(wide, left_on="flow_idx", right_index=True, how="left")
        nobs = inventory.groupby("flow_idx").size().rename("n_observations")
        out = out.merge(nobs, left_on="flow_idx", right_index=True, how="left")
        out["n_observations"] = out["n_observations"].fillna(0).astype(int)

    if attribution is not None:
        keep = ["flow_idx", "determined_by", "prior_mean", "prior_sd",
                "contraction", "shift_in_prior_sd"]
        keep = [c for c in keep if c in attribution.columns]
        keep += [c for c in attribution.columns if c.startswith("share_")]
        out = out.merge(attribution[keep], on="flow_idx", how="left")

    return out


def build_diagnostics_table(run_dir: str) -> pd.DataFrame:
    conv = _read(run_dir, "convergence_summary.csv")
    if conv is None:
        raise FileNotFoundError(f"{run_dir}: convergence_summary.csv not found")
    return conv


def build_node_balance_table(run_dir: str) -> pd.DataFrame | None:
    return _read(run_dir, "node_balance_diagnostics.csv")


def summarize_run(run_dir: str) -> dict:
    """Run-level numbers for the diagnostics narrative."""
    conv = _read(run_dir, "convergence_summary.csv")
    nb = _read(run_dir, "node_balance_diagnostics.csv")
    samples = np.load(long_path(os.path.join(run_dir, "posterior_samples.npy")))
    attribution = _read(run_dir, "attribution.csv")

    row = {"run": _label(run_dir), "n_flows": samples.shape[1], "n_draws": samples.shape[0]}
    if conv is not None:
        row.update({
            "rhat_median": conv["rhat"].median(),
            "rhat_max": conv["rhat"].max(),
            "ess_min": conv["ess"].min(),
            "ess_median": conv["ess"].median(),
            "n_flagged": int((~conv["converged"]).sum()),
        })
    if nb is not None and "residual_pct_mean" in nb.columns:
        row.update({
            "mb_residual_kt_max": nb["residual_mean"].abs().max(),
            "mb_residual_pct_max": nb["residual_pct_mean"].abs().max(),
            "mb_residual_pct_median": nb["residual_pct_mean"].abs().median(),
        })
    if attribution is not None and "determined_by" in attribution.columns:
        for k, v in attribution["determined_by"].value_counts().items():
            row[f"flows_{k.replace('-', '_')}"] = int(v)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories written by the pipeline")
    ap.add_argument("--out-dir", default="runs/si_tables")
    ap.add_argument("--ci-level", type=float, default=0.95)
    args = ap.parse_args(argv)

    summaries = []
    for run in args.runs:
        tag = _label(run)
        rec = build_reconciliation_table(run, ci_level=args.ci_level)
        rec.to_csv(prepare_output(os.path.join(args.out_dir, f"{tag}_reconciliation.csv")), index=False)

        diag = build_diagnostics_table(run)
        diag.to_csv(prepare_output(os.path.join(args.out_dir, f"{tag}_diagnostics.csv")), index=False)

        nb = build_node_balance_table(run)
        if nb is not None:
            nb.to_csv(prepare_output(os.path.join(args.out_dir, f"{tag}_node_balance.csv")), index=False)

        inv = _read(run, "source_inventory.csv")
        if inv is not None:
            inv.to_csv(prepare_output(os.path.join(args.out_dir, f"{tag}_source_inventory.csv")), index=False)

        # Sources reporting at coarser resolution than the model appear here
        # rather than in the per-flow table, since they constrain sums.
        agg = _read(run, "aggregate_observations.csv")
        if agg is not None:
            agg.to_csv(prepare_output(os.path.join(args.out_dir, f"{tag}_aggregate_observations.csv")), index=False)

        summaries.append(summarize_run(run))
        print(f"[{tag}] {len(rec)} flows; "
              f"observed {int((rec.get('n_observations', pd.Series(0)) > 0).sum())}, "
              f"multi-source {int((rec.get('n_observations', pd.Series(0)) > 1).sum())}")

    summary = pd.DataFrame(summaries)
    path = os.path.join(args.out_dir, "run_summary.csv")
    summary.to_csv(prepare_output(path), index=False)
    print(f"\nSI tables written to {args.out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
