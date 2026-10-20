# How to run

## 0. Install

```bash
pip install -e .
# or: conda env create -f environment.yml
```

Dependencies: numpy, pandas, scipy, pyyaml, openpyxl, plotly, kaleido.

## 1. Validate the 2017 source (pre-flight check)

```bash
python scripts/validate_2017_source.py configs/aluminum_2017.yaml
```

Fails if `data.source_name` is not a recognized 2017 table. The default config
points at the `"2017 data"` sheet of `data/raw/aluminum/US aluminum flows.xlsx`.

## 2. Run the 2017 baseline

```bash
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2017.yaml
```

This will:

1. Load the `"2017 data"` sheet, using `USGS value 1` as observations and the
   three pedigree columns to derive per-observation sigma.
2. Build the mass-balance system — 233 flows, 62 internal-node constraints.
3. Solve the constrained MAP problem.
4. Sample the posterior with the truncated-normal Gibbs sampler, and compute
   split-R̂ and ESS for every flow.
5. Write outputs to `runs/aluminum_2017_baseline/`.
6. Export the 2017 posterior as the 2022 prior, and the node allocation ratios
   as transferable soft constraints, into `data/canonical/aluminum/`.

The 2017 run must be executed before 2022, which depends on both exports.

## 3. Run the 2022 update

```bash
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2022.yaml
```

Adds the transferred prior and the allocation-ratio soft constraints on top of
the same machinery. Mass balance and bounds remain hard constraints; the
allocation ratios are relaxed, with strength set by
`allocation_constraints.relaxation`.

## 4. Outputs

Written to `runs/<run_name>/`:

| File | Contents |
|---|---|
| `posterior_mean.csv` | per-flow MAP estimate, posterior mean, and bounds, alongside the input table |
| `posterior_samples.npy` | `(n_chains·n_draws, n_flows)` posterior draws — the source of all credible intervals |
| `convergence_summary.csv` | per-flow split-R̂, ESS, mean, sd, 95% interval, and a `converged` flag |
| `node_balance_diagnostics.csv` | per-node posterior residual (inflow − outflow) summary |
| `allocation_shares.csv` | (2017) per-node allocation shares and their posterior sd |
| `allocation_residuals.csv` | (2022) realized deviation from each transferred ratio, in share units |

**Check the convergence summary before using any interval.** Flows failing
R̂ ≤ 1.05 or ESS ≥ 400 are flagged in a warning at the end of the run; their
credible intervals are not reliable.

## 5. Sensitivity analyses

Two assumptions materially affect the results and each has a sweep:

```bash
# pedigree -> observation-sigma mapping (2017)
python scripts/run_pedigree_sensitivity.py

# strength of the 2017 -> 2022 allocation-ratio constraints
python scripts/run_allocation_sensitivity.py
```

Each writes a `comparison.csv` under `runs/`. Both should be run before
reporting, since neither assumption is a fitted quantity.

## 6. Figures

```bash
python scripts/plot_sankey_ci.py --run-dir runs/aluminum_2022_update
```

Produces Sankey diagrams with links colored by credible-interval width, in
absolute and relative modes.

## Notes

- 2017 and 2022 share an identical 233-flow index and an identical node set, so
  the two vintages are directly comparable flow-for-flow.
- Runtime is dominated by sampling: roughly 10–30 minutes per year at the
  default 4 chains × 2000 draws with thinning.
- The pedigree → sigma mapping is the largest unverified assumption in the
  pipeline; see `docs/pedigree_sensitivity.md`.
