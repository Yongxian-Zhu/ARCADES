#!/usr/bin/env python3
"""
scripts/plot_sankey_ci.py
---------------------------
CLI entrypoint: read a completed pipeline run's posterior outputs and
produce absolute + relative credible-interval Sankey diagrams.

See arcade_mfa_aluminum.plots.sankey_ci module docstring for the full
adaptation notes (Lupton & Allwood / bayesian-mfa-mcmc trace_sankey_helpers.py
-> this package's flow-based posterior draws).

Usage
-----
Run both figures (absolute + relative) for a completed run:
    python scripts/plot_sankey_ci.py --run runs/aluminum_2017_baseline

Run only one:
    python scripts/plot_sankey_ci.py --run runs/aluminum_2017_baseline --mode relative

Common options:
    --ci-level 0.90            # use a 90% CI instead of the 95% default
    --units "Mt/y" --unit-scale 0.001   # display in Mt/y (raw data is kt/y)
    --min-flow 0.5              # drop flows with |median| below this (decluttering)
    --colorscale Viridis        # any Plotly named colorscale
    --workbook data/raw/aluminum/US aluminum flows.xlsx   # for clean node_catalog labels
    --formats pdf svg png html  # which export formats to attempt

Expects the run directory to contain (as written by `pipeline.run()`):
    posterior_mean.csv
    posterior_samples.npy
Figures are named after the run directory.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running directly from the scripts/ dir without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arcade_mfa_aluminum.paths import long_path  # noqa: E402
from arcade_mfa_aluminum.plots.sankey_ci import (  # noqa: E402
    build_posterior_flow_summary,
    make_sankey_figure,
    save_figure,
)


def _run_label(run_dir: str, explicit: str | None) -> str:
    """Label used to name the output figures; defaults to the run directory."""
    if explicit:
        return explicit
    mean_csv = os.path.join(run_dir, "posterior_mean.csv")
    if not os.path.exists(long_path(mean_csv)):
        raise FileNotFoundError(
            f"No posterior_mean.csv in {run_dir}. "
            f"Has `python -m arcade_mfa_aluminum.cli run --config ...` been run yet?"
        )
    return os.path.basename(os.path.normpath(run_dir))

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run directory, e.g. runs/aluminum_2017_baseline")
    ap.add_argument("--run-name", default=None, help="Override inferred run name (file stem).")
    ap.add_argument("--out", default=None, help="Output dir for figures (default: <run>/plots).")
    ap.add_argument("--mode", choices=["absolute", "relative", "both"], default="both")
    ap.add_argument("--ci-level", type=float, default=0.95, help="Central CI width, e.g. 0.95, 0.90.")
    ap.add_argument("--units", default="kt/y", help="Display units label.")
    ap.add_argument("--unit-scale", type=float, default=1.0, help="Multiply raw flow values for display.")
    ap.add_argument("--min-flow", type=float, default=0.0, help="Drop |median| flows below this threshold.")
    ap.add_argument("--colorscale", default="YlOrRd", help="Plotly named colorscale.")
    ap.add_argument("--workbook", default=None,
                     help="Path to US aluminum flows.xlsx for clean node_catalog labels (optional).")
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png", "html"],
                     help="Export formats to attempt (pdf/svg/png need `kaleido`).")
    args = ap.parse_args(argv)

    run_name = _run_label(args.run, args.run_name)
    mean_csv = os.path.join(args.run, "posterior_mean.csv")
    samples_npy = os.path.join(args.run, "posterior_samples.npy")
    out_dir = args.out or os.path.join(args.run, "plots")

    print(f"[plot_sankey_ci] run_name={run_name}")
    print(f"[plot_sankey_ci] reading posterior mean table -> {mean_csv}")
    print(f"[plot_sankey_ci] reading posterior samples     -> {samples_npy}")

    summary = build_posterior_flow_summary(
        long_path(mean_csv), long_path(samples_npy),
        ci_level=args.ci_level,
        workbook_path=args.workbook,
        units=args.units,
        unit_scale=args.unit_scale,
    )
    print(f"[plot_sankey_ci] {len(summary.df)} flows loaded; "
          f"{int(args.ci_level * 100)}% CI computed from posterior draws.")

    modes = ["absolute", "relative"] if args.mode == "both" else [args.mode]
    for mode in modes:
        fig = make_sankey_figure(
            summary, mode=mode, min_flow=args.min_flow, colorscale=args.colorscale
        )
        out_prefix = os.path.join(out_dir, f"sankey_{mode}_ci")
        written = save_figure(fig, out_prefix, formats=args.formats)
        print(f"[plot_sankey_ci] {mode} CI Sankey -> {list(written.values())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
