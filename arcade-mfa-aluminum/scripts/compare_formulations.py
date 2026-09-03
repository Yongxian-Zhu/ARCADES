"""
scripts/compare_formulations.py
-------------------------------
Compare the flow-based and node-allocation reconciliations.

The two formulations are reparameterisations of the same network -- node
throughputs times allocation shares reproduce the flows exactly -- but they do
not carry the same likelihood. The flow formulation places an independent normal
term on each observed flow; the node-allocation formulation places a normal term
on each fully observed node total and a Dirichlet term on its split, with the
concentration set by pedigree (SI Equation S17).

So this is a genuine cross-check rather than an identity. Close agreement means
the reconciliation is driven by the data and the network rather than by the
choice of parameterisation. Disagreement localises where the answer depends on
how the uncertainty was expressed, which is a result worth reporting rather
than a defect to tune away.

Usage:
    python scripts/compare_formulations.py
    python scripts/compare_formulations.py --years 2017 --min-flow 50
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.paths import long_path, prepare_output

#: year -> (flow-based run dir, node-allocation run dir)
DEFAULT_RUNS = {
    "2017": ("runs/aluminum_2017_baseline", "runs/aluminum_2017_node_allocation"),
    "2022": ("runs/aluminum_2022_update", "runs/aluminum_2022_node_allocation"),
}


def _load(run_dir: str):
    mean_csv = long_path(os.path.join(run_dir, "posterior_mean.csv"))
    samples = long_path(os.path.join(run_dir, "posterior_samples.npy"))
    if not os.path.exists(mean_csv):
        raise FileNotFoundError(f"{run_dir}: posterior_mean.csv not found -- run it first")
    df = pd.read_csv(mean_csv)
    S = np.load(samples) if os.path.exists(samples) else None
    if S is not None and S.ndim == 3:                    # (chains, draws, flows)
        S = S.reshape(-1, S.shape[-1])
    return df, S


def compare_year(year: str, flow_dir: str, na_dir: str, min_flow: float) -> pd.DataFrame:
    f_df, f_S = _load(flow_dir)
    n_df, n_S = _load(na_dir)
    if len(f_df) != len(n_df):
        raise ValueError(f"{year}: {len(f_df)} flows vs {len(n_df)} -- different networks")

    def summarise(df, S):
        if S is not None:
            lo, med, hi = np.percentile(S, [2.5, 50, 97.5], axis=0)
        else:
            med = df["posterior_mean"].to_numpy()
            lo = df.get("lower_bound", pd.Series(med)).to_numpy()
            hi = df.get("upper_bound", pd.Series(med)).to_numpy()
        return med, lo, hi

    f_med, f_lo, f_hi = summarise(f_df, f_S)
    n_med, n_lo, n_hi = summarise(n_df, n_S)

    denom = np.maximum(np.abs(f_med), 1e-9)
    out = pd.DataFrame({
        "year": year,
        "flow_idx": np.arange(len(f_df)),
        "from_node": f_df["from_node_name"],
        "to_node": f_df["to_node_name"],
        "flow_median": f_med, "flow_ci_lo": f_lo, "flow_ci_hi": f_hi,
        "nodealloc_median": n_med, "nodealloc_ci_lo": n_lo, "nodealloc_ci_hi": n_hi,
        "abs_diff": n_med - f_med,
        "rel_diff": (n_med - f_med) / denom,
        "flow_ci_width": f_hi - f_lo,
        "nodealloc_ci_width": n_hi - n_lo,
    })
    # Do the intervals overlap? Two formulations disagreeing by more than their
    # own stated uncertainty is a stronger statement than a percentage gap.
    out["intervals_overlap"] = (out.flow_ci_lo <= out.nodealloc_ci_hi) & \
                               (out.nodealloc_ci_lo <= out.flow_ci_hi)
    out["material"] = out.flow_median >= min_flow
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--min-flow", type=float, default=100.0,
                    help="threshold for the headline statistics (kt/y)")
    ap.add_argument("--out", default="runs/si_tables/formulation_comparison.csv")
    args = ap.parse_args(argv)

    frames = []
    for y in args.years:
        if y not in DEFAULT_RUNS:
            raise SystemExit(f"unknown year {y}; expected one of {list(DEFAULT_RUNS)}")
        frames.append(compare_year(y, *DEFAULT_RUNS[y], min_flow=args.min_flow))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(prepare_output(args.out), index=False)

    print("=" * 78)
    print("FLOW-BASED vs NODE-ALLOCATION")
    print("=" * 78)
    for y, g in all_df.groupby("year"):
        m = g[g.material]
        r = m.rel_diff.abs()
        print(f"\n{y}: {len(g)} flows, {len(m)} at or above {args.min_flow:g} kt/y")
        print(f"  median |rel diff| {r.median():7.4f}   90th {r.quantile(0.9):7.4f}   "
              f"max {r.max():7.4f}")
        print(f"  within  5%: {int((r <= 0.05).sum()):3d}/{len(m)}"
              f"   within 10%: {int((r <= 0.10).sum()):3d}/{len(m)}"
              f"   within 25%: {int((r <= 0.25).sum()):3d}/{len(m)}")
        print(f"  credible intervals overlap: {int(m.intervals_overlap.sum())}/{len(m)}")
        print(f"  median CI width  flow {m.flow_ci_width.median():9,.1f} kt   "
              f"node-allocation {m.nodealloc_ci_width.median():9,.1f} kt")
        worst = m.reindex(m.rel_diff.abs().sort_values(ascending=False).index).head(6)
        print("  largest disagreements:")
        for _, x in worst.iterrows():
            print(f"    {str(x.from_node)[:26]:28s}->{str(x.to_node)[:22]:24s} "
                  f"flow {x.flow_median:9,.1f}   node-alloc {x.nodealloc_median:9,.1f}   "
                  f"{100*x.rel_diff:+7.1f}%")

    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
