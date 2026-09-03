#!/usr/bin/env python3
"""
arcade_mfa_aluminum.cli
------------------------
Command-line entrypoint for the standalone aluminum MFA package.

Two formulations of the same reconciliation are available. `run` uses the
flow-based formulation, where the flow magnitudes are the variables;
`run-node-allocation` uses the node-allocation formulation, where node
throughputs and the allocation shares that split them are the variables and the
pedigree score sets a Dirichlet concentration on each split.

Usage:
    python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2017.yaml
    python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2022.yaml

    python -m arcade_mfa_aluminum.cli run-node-allocation \\
        --config configs/aluminum_2017_node_allocation.yaml
"""

from __future__ import annotations

import argparse
import sys

from arcade_mfa_aluminum.pipeline import run as run_pipeline
from arcade_mfa_aluminum.pipeline_node_allocation import run as run_node_allocation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arcade-mfa")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run", help="Run the flow-based MFA reconciliation pipeline.")
    run_parser.add_argument("--config", required=True, help="Path to a run YAML config.")

    na_parser = sub.add_parser(
        "run-node-allocation",
        help="Run the node-allocation reconciliation (node totals and Dirichlet splits).")
    na_parser.add_argument("--config", required=True, help="Path to a run YAML config.")

    args = parser.parse_args(argv)

    if args.command == "run":
        outputs = run_pipeline(args.config)
        print(f"\nDone. Outputs written under: {outputs.run_dir}")
        print(f"Posterior mean table: {outputs.posterior_mean_csv_path}")
        return 0

    if args.command == "run-node-allocation":
        outputs = run_node_allocation(args.config)
        print(f"\nDone. Outputs written under: {outputs.run_dir}")
        print(f"Posterior mean table: {outputs.posterior_mean_csv_path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
