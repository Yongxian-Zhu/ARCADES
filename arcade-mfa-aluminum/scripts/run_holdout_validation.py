"""
scripts/run_holdout_validation.py
---------------------------------
Leave-one-source-out validation of the reconciliation.

Convergence diagnostics show the sampler explores the posterior properly. They
say nothing about whether the posterior is *right*. This script withholds one
data source at a time, refits, and scores the withheld values as genuine
out-of-sample predictions.

The question it answers is the one R2-7 asks: when the network structure and the
remaining sources are all the model has, does it recover a source it never saw?

A source is dropped from the likelihood entirely rather than down-weighted --
a large sigma still pulls the posterior, which would let the model peek at the
answer it is being scored against.

Three numbers are reported per withheld source:

`coverage_95`
    Fraction of withheld values falling inside the posterior 95% credible
    interval. Near 0.95 means the intervals are honest. Well below means
    overconfident; well above means uninformative.

`median_abs_pct_error`
    Typical size of the miss, relative to the withheld value.

`median_abs_z`
    Miss measured in units of the withheld value's own sigma, so a source
    reporting a quantity precisely is judged more strictly than one that does
    not.

Aggregate sources (the Aluminum Association, which reports totals spanning
several flows) are scored the same way against the reconciled sums.

Usage:
    python scripts/run_holdout_validation.py --config configs/aluminum_2017.yaml
    python scripts/run_holdout_validation.py --config configs/aluminum_2022.yaml --draws 500
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import pandas as pd
import yaml

from arcade_mfa_aluminum.paths import ensure_dir, long_path, prepare_output
from arcade_mfa_aluminum.pipeline import run as run_pipeline


def _load_aggregate_specs(cfg: dict) -> list[dict]:
    specs = list(cfg.get("aggregate_observations", []) or [])
    for extra in (cfg.get("aggregate_observation_files", []) or []):
        with open(long_path(extra)) as f:
            specs.extend((yaml.safe_load(f) or {}).get("aggregate_observations", []) or [])
    return specs


#: Withheld values below this magnitude are excluded from percentage-error
#: statistics. A relative error against a value of 0.001 kt/y is arithmetic
#: noise, not a prediction failure, and one such row otherwise dominates a mean.
PCT_ERROR_FLOOR_KT = 1.0

#: A hold-out that removes most of the evidence is not a test of prediction --
#: the model has nothing left to predict from. Such cases are reported but
#: flagged, so they are not read as a validation result.
MAX_DIAGNOSTIC_WITHHELD_FRACTION = 0.5


def _score(values, sigmas, med, lo, hi, n_total=None) -> dict:
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    med, lo, hi = map(lambda a: np.asarray(a, dtype=float), (med, lo, hi))
    inside = (values >= lo) & (values <= hi)
    z = np.abs(med - values) / np.maximum(sigmas, 1e-9)

    big = np.abs(values) >= PCT_ERROR_FLOOR_KT
    if big.any():
        pct = 100.0 * np.abs(med[big] - values[big]) / np.abs(values[big])
        med_pct, mean_pct = float(np.median(pct)), float(np.mean(pct))
    else:
        med_pct = mean_pct = float("nan")

    out = {
        "n_withheld": int(values.size),
        "n_scored_pct": int(big.sum()),
        "coverage_95": float(inside.mean()),
        "median_abs_pct_error": med_pct,
        "mean_abs_pct_error": mean_pct,
        "median_abs_z": float(np.median(z)),
        "max_abs_z": float(np.max(z)),
    }
    if n_total:
        frac = values.size / float(n_total)
        out["frac_of_all_observations"] = frac
        out["diagnostic"] = bool(frac <= MAX_DIAGNOSTIC_WITHHELD_FRACTION)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/holdout_<config stem>")
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="withhold only these sources (default: every source found)")
    args = ap.parse_args(argv)

    with open(long_path(args.config)) as f:
        base_cfg = yaml.safe_load(f)
    stem = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out_dir or os.path.join("runs", f"holdout_{stem}")
    ensure_dir(out_dir)

    # ---- baseline: every source in, and the inventory of what exists --------
    def make_cfg(name: str, disabled: list[str]) -> str:
        cfg = copy.deepcopy(base_cfg)
        cfg["run"]["name"] = f"holdout_{name}"
        cfg["diagnostics"]["output_dir"] = os.path.join(out_dir, name)
        cfg["output"]["posterior_mean_csv"] = os.path.join(out_dir, name, "posterior_mean.csv")
        cfg["output"]["posterior_samples_npy"] = os.path.join(out_dir, name, "posterior_samples.npy")
        cfg["output"]["save_prior_for_transfer"] = False
        cfg["output"]["save_allocation_shares"] = False
        src = dict(cfg.get("observation_sources") or {})
        for d in disabled:
            spec = dict(src.get(d) or {})
            spec["enabled"] = False
            src[d] = spec
        cfg["observation_sources"] = src
        if args.draws:
            cfg["inference"]["sampling"]["n_draws"] = args.draws
        if args.chains:
            cfg["inference"]["sampling"]["n_chains"] = args.chains
        path = os.path.join(out_dir, f"_config_{name}.yaml")
        with open(prepare_output(path), "w") as f:
            yaml.safe_dump(cfg, f)
        return path

    print("=== baseline fit (all sources) ===")
    base_out = run_pipeline(make_cfg("baseline", []))
    inventory = base_out.observations.copy()
    agg_specs = _load_aggregate_specs(base_cfg)

    flow_sources = sorted(inventory["source"].astype(str).unique())
    agg_sources = sorted({str(a.get("source", "")) for a in agg_specs} - {""})
    candidates = args.sources or sorted(set(flow_sources) | set(agg_sources))
    print(f"\nper-flow sources : {flow_sources}")
    print(f"aggregate sources: {agg_sources}")
    print(f"withholding      : {candidates}\n")

    rows, detail = [], []
    for src in candidates:
        tag = "".join(ch if ch.isalnum() else "_" for ch in src).strip("_").lower()
        held_flow = inventory[inventory["source"].astype(str) == src]
        held_agg = [a for a in agg_specs if str(a.get("source", "")) == src]
        if not len(held_flow) and not held_agg:
            print(f"--- skip {src}: nothing to withhold")
            continue

        print(f"=== withholding: {src} "
              f"({len(held_flow)} per-flow, {len(held_agg)} aggregate) ===")
        try:
            out = run_pipeline(make_cfg(tag, [src]))
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            rows.append({"withheld_source": src, "error": f"{type(exc).__name__}: {exc}"})
            continue

        S = out.posterior_samples
        lo_all, med_all, hi_all = np.percentile(S, [2.5, 50, 97.5], axis=0)

        if len(held_flow):
            idx = held_flow["flow_idx"].to_numpy(dtype=int)
            r = _score(held_flow["value"].to_numpy(), held_flow["sigma"].to_numpy(),
                       med_all[idx], lo_all[idx], hi_all[idx], n_total=len(inventory))
            r.update({"withheld_source": src, "kind": "per-flow"})
            rows.append(r)
            for j, (_, o) in zip(idx, held_flow.iterrows()):
                detail.append({
                    "withheld_source": src, "kind": "per-flow", "flow_idx": int(j),
                    "withheld_value": float(o["value"]), "sigma": float(o["sigma"]),
                    "posterior_median": float(med_all[j]),
                    "ci_lo": float(lo_all[j]), "ci_hi": float(hi_all[j]),
                    "inside_95": bool(lo_all[j] <= o["value"] <= hi_all[j]),
                })

        if held_agg:
            vals, sgs, meds, los, his = [], [], [], [], []
            for a in held_agg:
                f = np.asarray(a["flows"], dtype=int)
                draws = S[:, f].sum(axis=1)
                l, m, h = np.percentile(draws, [2.5, 50, 97.5])
                v = float(a["value"])
                sg = float(a["sigma"]) if "sigma" in a else \
                    float(a.get("relative_sigma", 0.10)) * max(abs(v), 1e-6)
                vals.append(v); sgs.append(sg); meds.append(m); los.append(l); his.append(h)
                detail.append({
                    "withheld_source": src, "kind": "aggregate", "name": a.get("name", ""),
                    "withheld_value": v, "sigma": sg, "posterior_median": m,
                    "ci_lo": l, "ci_hi": h, "inside_95": bool(l <= v <= h),
                })
            r = _score(vals, sgs, meds, los, his)
            r.update({"withheld_source": src, "kind": "aggregate"})
            rows.append(r)

    summary = pd.DataFrame(rows)
    summary.to_csv(prepare_output(os.path.join(out_dir, "holdout_summary.csv")), index=False)
    pd.DataFrame(detail).to_csv(
        prepare_output(os.path.join(out_dir, "holdout_detail.csv")), index=False)

    print(f"\n=== leave-one-source-out validation: {stem} ===")
    cols = [c for c in ["withheld_source", "kind", "n_withheld", "frac_of_all_observations",
                        "diagnostic", "coverage_95", "median_abs_pct_error",
                        "median_abs_z", "max_abs_z", "error"]
            if c in summary.columns]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    if "diagnostic" in summary.columns and (summary["diagnostic"] == False).any():  # noqa: E712
        bad = summary.loc[summary["diagnostic"] == False, "withheld_source"].tolist()  # noqa: E712
        print(f"\nNOT DIAGNOSTIC: {bad} — withholding these removes more than "
              f"{int(100 * MAX_DIAGNOSTIC_WITHHELD_FRACTION)}% of all observations, so the "
              f"model has little left to predict from. Report as a dependence "
              f"measure, not as validation.")
    print(f"\nwritten to {out_dir}/holdout_summary.csv and holdout_detail.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
