"""
scripts/run_robustness_suite.py
-------------------------------
Robustness of the reported conclusions to modelling assumptions.

Sampler convergence shows that the inference is computationally stable; it says
nothing about whether a conclusion depends on a judgment call. This script
re-runs the reconciliation under alternative assumptions -- prior strength,
pedigree-to-uncertainty mapping, mass-balance tolerance, routing-prior strength,
source weighting -- and reports the circularity indicators the paper claims,
each with a credible interval.

The output is deliberately framed around conclusions rather than flows. A
conclusion that holds across every case can be stated plainly; one that does not
must be reported as assumption-dependent, with the dependence shown.

Usage:
    python scripts/run_robustness_suite.py
    python scripts/run_robustness_suite.py --cases baseline prior_disabled --draws 400
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import pandas as pd
import yaml

from arcade_mfa_aluminum.aggregates import compute_aggregates
from arcade_mfa_aluminum.paths import ensure_dir, long_path, prepare_output
from arcade_mfa_aluminum.pipeline import run as run_pipeline


def deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


#: Indicators reported for every case. These are the quantities the conclusions
#: are stated in, so a reader can see directly whether a finding moves.
REPORTED = [
    "domestic_retention_rate",
    "eol_collection_rate",
    "eol_scrap_collected_total",
    "eol_scrap_recycled",
    "eol_scrap_exported",
    "semis_ingot_total",
    "secondary_metal_total",
    "domestic_consumption",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-config", default="configs/robustness_suite.yaml")
    ap.add_argument("--out-dir", default="runs/robustness")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="run only these case names (default: all)")
    ap.add_argument("--draws", type=int, default=None,
                    help="override n_draws for a faster exploratory pass")
    ap.add_argument("--chains", type=int, default=None)
    args = ap.parse_args(argv)

    with open(long_path(args.suite_config)) as f:
        suite = yaml.safe_load(f)
    with open(long_path(suite["base_config"])) as f:
        base_cfg = yaml.safe_load(f)

    ensure_dir(args.out_dir)
    cases = suite["cases"]
    if args.cases:
        cases = [c for c in cases if c["name"] in args.cases]
        if not cases:
            raise SystemExit(f"no cases matched {args.cases}")

    rows, failures = [], []
    for case in cases:
        name = case["name"]
        cfg = deep_update(base_cfg, case.get("overrides", {}))
        cfg["run"]["name"] = f"robustness_{name}"
        cfg["diagnostics"]["output_dir"] = os.path.join(args.out_dir, name)
        cfg["output"]["posterior_mean_csv"] = os.path.join(args.out_dir, name, "posterior_mean.csv")
        cfg["output"]["posterior_samples_npy"] = os.path.join(args.out_dir, name, "posterior_samples.npy")
        cfg["output"]["save_prior_for_transfer"] = False
        cfg["output"]["save_allocation_shares"] = False
        if args.draws:
            cfg["inference"]["sampling"]["n_draws"] = args.draws
        if args.chains:
            cfg["inference"]["sampling"]["n_chains"] = args.chains

        case_cfg = os.path.join(args.out_dir, f"_config_{name}.yaml")
        with open(prepare_output(case_cfg), "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n=== case: {name}  {case.get('overrides', {})} ===")
        try:
            out = run_pipeline(case_cfg)
        except Exception as exc:                      # a case may be infeasible
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failures.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        S = out.posterior_samples
        flows = pd.read_csv(long_path(os.path.join(args.out_dir, name, "posterior_mean.csv")))
        agg = compute_aggregates(S, flows).set_index("quantity")

        row = {"case": name, "overrides": str(case.get("overrides", {}))}
        conv = out.convergence_summary
        if len(conv):
            row["rhat_max"] = conv["rhat"].max()
            row["ess_min"] = conv["ess"].min()
            row["n_flagged"] = int((~conv["converged"]).sum())
        nb = out.node_balance_diagnostics
        if len(nb) and "residual_pct_mean" in nb.columns:
            row["mb_residual_pct_max"] = nb["residual_pct_mean"].abs().max()
        for q in REPORTED:
            if q in agg.index:
                row[q] = agg.loc[q, "median"]
                row[f"{q}_lo"] = agg.loc[q, "ci_lo"]
                row[f"{q}_hi"] = agg.loc[q, "ci_hi"]
        rows.append(row)

    if not rows:
        raise SystemExit("no case completed successfully")

    comp = pd.DataFrame(rows)
    comp.to_csv(prepare_output(os.path.join(args.out_dir, "comparison.csv")), index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            prepare_output(os.path.join(args.out_dir, "failures.csv")), index=False)

    # A case whose indicators are bit-identical to baseline did not perturb
    # anything: the knob it turns is inert for this configuration. Reporting it
    # as agreement would count an untested assumption as a robust one, which is
    # the opposite of what this suite is for.
    #
    # The usual cause is a knob that acts on data the target year does not
    # carry -- the pedigree mapping does nothing in a year whose observations
    # have no pedigree scores, and the zero-flow bound does nothing in a year
    # with no zero-valued observations. Such assumptions still matter, but they
    # have to be exercised against the year that carries the data.
    base = comp[comp.case == "baseline"]
    inert = []
    if len(base):
        cols = [q for q in REPORTED if q in comp.columns]
        for _, row in comp.iterrows():
            if row["case"] == "baseline":
                continue
            same = all(
                (pd.isna(row[q]) and pd.isna(base[q].iloc[0]))
                or np.isclose(row[q], base[q].iloc[0], rtol=0, atol=1e-9)
                for q in cols
            )
            if same:
                inert.append(row["case"])
    comp["exercised"] = ~comp["case"].isin(inert)
    comp.to_csv(prepare_output(os.path.join(args.out_dir, "comparison.csv")), index=False)

    print("\n" + "=" * 78)
    print("ROBUSTNESS SUMMARY")
    print("=" * 78)
    if inert:
        print(f"\nNOT EXERCISED ({len(inert)}): {', '.join(inert)}")
        print("  These reproduced the baseline exactly, so the assumption they vary has")
        print("  no effect on this configuration and is UNTESTED here -- do not report")
        print("  it as robust. Exercise it against a configuration whose data it acts on.")
        print(f"  Cases actually exercised: {len(comp) - len(inert) - 1} of {len(comp) - 1}.")
    show = ["case", "domestic_retention_rate", "eol_collection_rate",
            "eol_scrap_exported", "semis_ingot_total", "rhat_max", "n_flagged"]
    show = [c for c in show if c in comp.columns]
    print(comp[show].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    exercised = comp[comp["exercised"] | (comp["case"] == "baseline")]
    print("\nspread across EXERCISED cases (inert cases and failures excluded):")
    for q in REPORTED:
        if q in comp.columns:
            v = exercised[q].dropna()
            if len(v):
                b = base[q].iloc[0] if len(base) and q in base else np.nan
                print(f"  {q:28s} baseline {b:>10,.3f}   range [{v.min():,.3f}, {v.max():,.3f}]"
                      f"   spread {100 * (v.max() - v.min()) / max(abs(v.median()), 1e-9):5.1f}% of median")

    if "domestic_retention_rate" in comp.columns:
        r = exercised["domestic_retention_rate"].dropna()
        print(f"\n  domestic retention stays below 60% in {int((r < 0.60).sum())}/{len(r)} cases"
              f"  -> the retention constraint is {'robust' if (r < 0.60).all() else 'NOT robust'}")
    if failures:
        print(f"\n  {len(failures)} case(s) failed; see failures.csv")
    print(f"\nwritten to {args.out_dir}/comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
