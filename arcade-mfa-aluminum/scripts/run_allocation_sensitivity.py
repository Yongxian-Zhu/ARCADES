"""
scripts/run_allocation_sensitivity.py
-------------------------------------
Sensitivity sweep over the strength of the 2017 -> 2022 allocation-ratio soft
constraints (see src/arcade_mfa_aluminum/allocation.py and
configs/allocation_sensitivity.yaml).

For each case in the sweep config this script:
  1. Builds a modified copy of the 2022 config with that case's
     `allocation_constraints` overrides applied.
  2. Runs the full 2022 pipeline (ingest -> mass balance -> MAP with the soft
     allocation block -> truncated-Gibbs sampling -> diagnostics).
  3. Collects the numbers that decide whether the constraint strength is
     defensible:
       - how many flows still fail convergence, and how much posterior
         variance they hold (the identifiability problem this feature targets)
       - mean 95% CI width (tighter constraints should narrow intervals)
       - max |z| of the allocation residuals (how hard 2022 data fights the
         2017 split -- the masking risk)
       - medians for the flows that were unidentified before the transfer
  4. Writes one comparison CSV.

Reading the output: a good setting narrows the intervals and resolves the
unidentified flows WITHOUT driving allocation residuals far beyond their
tolerance. If a case shows both narrow CIs and large |z|, it is overriding the
2022 data rather than informing it.

Usage:
    python scripts/run_allocation_sensitivity.py \
        --sweep-config configs/allocation_sensitivity.yaml \
        --base-config configs/aluminum_2022.yaml \
        --key-flows 146 206 215 216 229 231

Requires the 2017 run to have been executed first (it writes the shares .npz).
Each case is a full MAP + sampling cycle, so wall-clock scales linearly with
the number of cases.
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


def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-config", default="configs/allocation_sensitivity.yaml")
    ap.add_argument("--base-config", default="configs/aluminum_2022.yaml")
    ap.add_argument("--key-flows", type=int, nargs="*",
                    default=[146, 206, 215, 216, 229, 231],
                    help="flows unidentified in the no-allocation baseline")
    ap.add_argument("--out-dir", default="runs/allocation_sensitivity")
    args = ap.parse_args()

    with open(long_path(args.sweep_config)) as f:
        sweep = yaml.safe_load(f)
    with open(long_path(args.base_config)) as f:
        base_cfg = yaml.safe_load(f)

    ensure_dir(args.out_dir)
    baseline = sweep["baseline"]

    rows = []
    for case in sweep["cases"]:
        name = case["name"]
        alloc = _deep_update(baseline, case.get("overrides", {}))

        cfg = copy.deepcopy(base_cfg)
        cfg["allocation_constraints"] = alloc
        cfg["run"]["name"] = f"allocation_{name}"
        cfg["diagnostics"]["output_dir"] = os.path.join(args.out_dir, name)
        cfg["output"]["posterior_mean_csv"] = os.path.join(
            args.out_dir, name, f"{name}_posterior_mean.csv")
        cfg["output"]["posterior_samples_npy"] = os.path.join(
            args.out_dir, name, f"{name}_posterior_samples.npy")

        case_cfg_path = os.path.join(args.out_dir, f"_config_{name}.yaml")
        with open(prepare_output(case_cfg_path), "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n=== allocation case: {name} ({case.get('overrides', {})}) ===")
        out = run_pipeline(case_cfg_path)

        S = out.posterior_samples
        lo, hi = np.percentile(S, [2.5, 97.5], axis=0)
        conv = out.convergence_summary
        sd = S.std(axis=0)

        if len(conv):
            bad = ~conv["converged"].to_numpy()
            var_share = float((sd[bad] ** 2).sum() / max((sd ** 2).sum(), 1e-30))
            row = {
                "case": name,
                "relaxation": alloc.get("relaxation") if alloc.get("enabled") else None,
                "min_share_sigma": alloc.get("min_share_sigma") if alloc.get("enabled") else None,
                "enabled": bool(alloc.get("enabled", False)),
                "n_unconverged": int(bad.sum()),
                "variance_share_unconverged": var_share,
                "rhat_max": float(conv["rhat"].max()),
                "ess_min": float(conv["ess"].min()),
            }
        else:
            row = {"case": name, "enabled": bool(alloc.get("enabled", False))}

        row["mean_95ci_width"] = float(np.mean(hi - lo))
        row["median_95ci_width"] = float(np.median(hi - lo))

        ar = out.allocation_residuals
        if len(ar):
            z = ar["z_vs_tolerance"].abs()
            row["alloc_max_abs_z"] = float(z.max())
            row["alloc_rows_beyond_2sigma"] = int((z > 2).sum())
            row["alloc_median_abs_share_dev"] = float(ar["share_dev_abs_mean"].median())
        else:
            row["alloc_max_abs_z"] = np.nan
            row["alloc_rows_beyond_2sigma"] = 0
            row["alloc_median_abs_share_dev"] = np.nan

        for i in args.key_flows:
            if i < S.shape[1]:
                row[f"flow_{i}_median"] = float(np.median(S[:, i]))
                row[f"flow_{i}_ci_width"] = float(hi[i] - lo[i])
        rows.append(row)

    comparison = pd.DataFrame(rows)
    path = os.path.join(args.out_dir, "comparison.csv")
    comparison.to_csv(prepare_output(path), index=False)
    print(f"\nSaved allocation sensitivity comparison -> {path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
