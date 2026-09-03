"""
scripts/run_pedigree_sensitivity.py
------------------------------------
Systematic sensitivity sweep over the pedigree(coverage/frequency/spatial
boundary) -> observation-sigma mapping (see
arcade_mfa_aluminum.adapters.aluminum.adapter.pedigree_to_rel_sigma and
docs/pedigree_sensitivity.md for the full review workflow).

For each case in configs/pedigree_mapping.yaml, this script:
  1. Builds a modified copy of the 2017 config with that case's mapping
     overrides applied to observation_sigma.pedigree_mapping.
  2. Runs the full 2017 flow_based pipeline (ingest -> mass balance -> MAP
     -> Laplace sampling) via arcade_mfa_aluminum.pipeline.run.
  3. Collects: MAP mass-balance residual (max abs), posterior interval width
     (mean over flows), and per-flow posterior median for a configurable set
     of "key flows" to track across cases.
  4. Writes a single comparison CSV to runs/pedigree_sensitivity/comparison.csv.

Usage:
    python scripts/run_pedigree_sensitivity.py \
        --sweep-config configs/pedigree_mapping.yaml \
        --base-config configs/aluminum_2017.yaml \
        --key-flows 0 1 2 10 50           # optional: flow_idx values to track

Runs a full MAP + sampling cycle per case, so wall-clock scales linearly with
the number of cases -- keep the sweep grid small (~10 cases) or reduce
inference.map.maxiter / inference.sampling.n_draws in the base config for a
faster exploratory pass before a final high-fidelity sweep.
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import pandas as pd
import yaml

from arcade_mfa_aluminum.paths import ensure_dir, prepare_output
from arcade_mfa_aluminum.pipeline import run as run_pipeline


def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-config", default="configs/pedigree_mapping.yaml")
    ap.add_argument("--base-config", default="configs/aluminum_2017.yaml")
    ap.add_argument("--key-flows", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", default="runs/pedigree_sensitivity")
    args = ap.parse_args()

    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)
    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    ensure_dir(args.out_dir)
    baseline_mapping = sweep["baseline"]

    rows = []
    for case in sweep["cases"]:
        case_name = case["name"]
        mapping = _deep_update(baseline_mapping, case.get("overrides", {}))

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("observation_sigma", {})["pedigree_mapping"] = mapping
        cfg["run"]["name"] = f"pedigree_sensitivity_{case_name}"
        cfg["diagnostics"]["output_dir"] = os.path.join(args.out_dir, case_name)
        cfg["output"]["posterior_mean_csv"] = os.path.join(
            args.out_dir, case_name, f"{case_name}_posterior_mean.csv"
        )
        cfg["output"]["posterior_samples_npy"] = os.path.join(
            args.out_dir, case_name, f"{case_name}_posterior_samples.npy"
        )
        # Don't let every sensitivity case overwrite the real transfer prior.
        cfg["output"]["save_prior_for_transfer"] = False

        case_cfg_path = os.path.join(args.out_dir, f"_config_{case_name}.yaml")
        with open(prepare_output(case_cfg_path), "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n=== Running case: {case_name} (mapping={mapping}) ===")
        outputs = run_pipeline(case_cfg_path)

        resid = outputs.node_balance_diagnostics
        # Column is "residual_mean" (see graph.posterior_node_balance_diagnostics),
        # not "mean_residual" -- the old spelling silently produced an all-NaN column.
        max_abs_node_resid = (
            resid["residual_mean"].abs().max() if "residual_mean" in resid.columns
            and len(resid) else np.nan
        )

        samples = outputs.posterior_samples  # (n_draws, n_flows)
        lo = np.percentile(samples, 2.5, axis=0)
        hi = np.percentile(samples, 97.5, axis=0)
        mean_interval_width = float(np.mean(hi - lo))

        row = {
            "case": case_name,
            "max_abs_node_residual": max_abs_node_resid,
            "mean_95pct_interval_width": mean_interval_width,
            **{f"flow_{i}_posterior_median": float(np.median(samples[:, i]))
               for i in args.key_flows if i < samples.shape[1]},
        }
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = os.path.join(args.out_dir, "comparison.csv")
    comparison.to_csv(prepare_output(comparison_path), index=False)
    print(f"\nSaved sensitivity comparison table -> {comparison_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
