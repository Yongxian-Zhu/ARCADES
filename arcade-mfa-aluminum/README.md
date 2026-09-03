# Bayesian material flow analysis of the U.S. aluminum cycle (2017, 2022)

Reproducible code and data for the aluminum material flow analysis reported in
the accompanying paper.

The package reconciles an aluminum flow network against observations under
mass-balance and bound constraints, quantifies the uncertainty of every flow,
and carries information from the 2017 baseline forward into the sparser 2022
update.

Every reported value enters the likelihood as its own term, so where sources
disagree the posterior settles between them by relative precision rather than by
which column was read first. Mass balance is applied as a penalty by default,
so closure residuals are finite, reported, and interpretable rather than driven
to machine epsilon.

**Network:** 234 flows, 105 nodes, 62 internal nodes carrying mass-balance
constraints.

## Method

| Stage | Approach |
|---|---|
| Ingestion | Single authoritative workbook; pedigree confidence scores mapped to per-observation sigma |
| Likelihood | One term per reported value; flows observed by several sources are reconciled by relative precision. Sources publishing only totals constrain a sum |
| Point estimate | Constrained quadratic MAP, `trust-constr` with an analytic Hessian, subject to soft mass balance and hard box bounds |
| Uncertainty | Truncated-normal Gibbs sampling with hit-and-run moves; every draw satisfies the bounds exactly, and mass balance exactly in `hard` mode |
| Convergence | Split-R̂ and ESS per flow on every run; non-converged flows flagged |
| 2017 → 2022 | Posterior transferred as a prior, plus node allocation ratios transferred as adjustable soft constraints |
| Attribution | Per-flow precision shares showing whether a value is set by observation, prior or network structure |
| Validation | Leave-one-source-out prediction of withheld observations |
| Reporting | Credible-interval Sankey diagrams; robustness suite over every principal assumption |

Full detail in [docs/model_formulation.md](docs/model_formulation.md).

## Quick start

```bash
pip install -e .

python scripts/validate_2017_source.py configs/aluminum_2017.yaml
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2017.yaml
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2022.yaml
```

The 2017 run must precede 2022, which consumes both of its exports. See
[docs/how_to_run.md](docs/how_to_run.md).

## Layout

```
configs/         run configurations (2017, 2022) and the two sensitivity grids
data/raw/        source workbook (authoritative input)
data/canonical/  derived artifacts: transfer prior, allocation shares
docs/            method, data schema, run instructions, sensitivity workflows
runs/            run outputs
scripts/         pre-flight validation, robustness suite, hold-out validation,
                 SI tables, figure generation
src/             the package
tests/           test suite
```

## Reproducing the published results

The 2017 run must precede 2022: it writes the transfer prior and the allocation
shares that 2022 consumes.

```bash
pip install -e .

# pre-flight check on the source workbook
python scripts/validate_2017_source.py configs/aluminum_2017.yaml

# the two reconciliations
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2017.yaml
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2022.yaml

# Supporting Information tables
python scripts/build_si_tables.py \
    --runs runs/aluminum_2017_baseline runs/aluminum_2022_update

# credible-interval Sankey diagrams, absolute and relative
python scripts/plot_sankey_ci.py --run runs/aluminum_2017_baseline --mode both \
    --workbook "data/raw/aluminum/US aluminum flows.xlsx"
python scripts/plot_sankey_ci.py --run runs/aluminum_2022_update --mode both \
    --workbook "data/raw/aluminum/US aluminum flows.xlsx"

# the same reconciliations under the node-allocation formulation (SI S15-S18),
# and a per-flow comparison of the two
python -m arcade_mfa_aluminum.cli run-node-allocation --config configs/aluminum_2017_node_allocation.yaml
python -m arcade_mfa_aluminum.cli run-node-allocation --config configs/aluminum_2022_node_allocation.yaml
python scripts/compare_formulations.py
# robustness of the conclusions to every principal assumption
python scripts/run_robustness_suite.py

# leave-one-source-out validation
python scripts/run_holdout_validation.py --config configs/aluminum_2017.yaml
python scripts/run_holdout_validation.py --config configs/aluminum_2022.yaml
```

Published outputs are in [`results/`](results/) for comparison; see
[`results/README.md`](results/README.md) for what each file contains. Runs write
to `runs/`, which is not tracked.

Sampling is seeded, so a rerun on the same platform reproduces the published
draws. Across platforms the medians and intervals should agree to well within
their own width, but the individual draws will differ.

## Package

| Module | Responsibility |
|---|---|
| `adapters/aluminum/` | Workbook ingestion; pedigree → relative sigma |
| `graph.py` | Mass-balance construction, nullspace, node-balance diagnostics |
| `inference/optimization.py` | Constrained quadratic MAP solver |
| `inference/sampling.py` | Truncated-normal Gibbs sampler (and the legacy Laplace sampler) |
| `transfer.py` | 2017 posterior → 2022 prior, with a numerically safe precision |
| `allocation.py` | Node allocation ratios → soft linear constraints |
| `diagnostics.py` | Split-R̂ and effective sample size |
| `priors/quality_sigma.py` | Pedigree scores → observation sigma |
| `plots/sankey_ci.py` | Credible-interval Sankey diagrams |
| `paths.py` | Long-path-safe filesystem helpers (Windows) |
| `pipeline.py`, `cli.py` | Orchestration and command-line entry point |

## Tests

```bash
python -m pytest tests -q
```

Coverage focuses on the properties that matter and cannot be checked by
inspection: that sampled draws reproduce the target covariance and the analytic
truncated normal, that every draw satisfies both constraint sets, that the
transferred precision is positive semi-definite despite a rank-deficient input,
that the convergence diagnostics fire on bad chains and stay quiet on good ones,
and that the allocation constraints are exact at their target ratio and respond
to the relaxation knob in both directions.

## Interpreting the results

Two points bear directly on how the numbers should be read.

**Check the convergence summary.** Every run writes per-flow R̂ and ESS. Flows
that fail are flagged in a warning; their credible intervals are not reliable.

**Some 2022 flows are weakly identified.** Only about a fifth of 2022 flows
carry an observation. For some of the remainder — particularly flows tied
together in pairs by a shared mass-balance constraint — the data determine the
sum but not the individual values, and the reported interval reflects the bound
heuristic rather than evidence. The allocation-ratio constraints exist to supply
that missing structure, and
[`scripts/run_allocation_sensitivity.py`](scripts/run_allocation_sensitivity.py)
quantifies how much of the result depends on that assumption.

Neither the pedigree → sigma mapping nor the allocation constraint strength is a
fitted quantity. Both have sweeps, and both should be run before reporting.

## Citation

If you use this code or data, please cite the accompanying paper:

> Yongxian Zhu, David Thierry, Barbara K. Reck, Madeleine Wahl, and Sarang Supekar.
> *Mapping Aluminum Flows in the United States using Bayesian Material Flow
> Analysis.* ChemRxiv, 19 February 2026.
> [doi:10.26434/chemrxiv.15000222/v1](https://doi.org/10.26434/chemrxiv.15000222/v1)

Preprint; this reference will be updated on journal publication.
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
