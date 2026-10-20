# Pedigree -> uncertainty mapping: review workflow and sensitivity analysis

## Why this matters
The authoritative `"2017 data"` sheet in `US aluminum flows.xlsx` carries
three confidence/pedigree dimensions per USGS value (coverage, frequency,
spatial boundary; small integers, e.g. 1/2/4 observed) instead of a single
composite quality score. **No codebook or legend for these columns was found
for these columns.** This package converts them into an observation
sigma (how much the MAP/sampler trusts vs. discounts each flow's reported
value relative to the mass-balance constraints) via an explicit, documented
assumption. Getting this wrong biases every downstream posterior estimate,
so it needs a systematic review, not a one-time guess.

## Where the mapping lives (read these two files first)
1. `src/arcade_mfa_aluminum/adapters/aluminum/adapter.py::pedigree_to_rel_sigma`
   — the actual combiner function. Docstring spells out every assumption.
2. `src/arcade_mfa_aluminum/priors/quality_sigma.py::sigma_obs_from_pedigree`
   — vectorized wrapper called from the pipeline; accepts a `mapping` dict
   that is forwarded straight to `pedigree_to_rel_sigma` as kwargs.

Both are pure functions with no hidden state — the entire mapping behavior
is controlled by the keyword arguments below (all exposed to config):

| knob | meaning | default | what to test |
|---|---|---|---|
| `scale_max` | assumed top of the raw 1..N pedigree scale | 4.0 | 3.0, 5.0 if scale assumption is wrong |
| `sigma_rel_best` | relative sigma at the best-scored pedigree value | 0.05 | tighter (0.02) / looser (0.10) |
| `sigma_rel_worst` | relative sigma at the worst-scored pedigree value | 1.0 | tighter (0.5) / looser (2.0) |
| `global_multiplier` | uniform scale factor on the final sigma | 1.0 | 0.5x, 2x, 4x — coarsest, cheapest check |
| `combiner` | how the 3 dimensions combine into one sigma | `geometric_mean` | `arithmetic_mean`, `max`, `weighted_mean` |
| `direction` | which end of the scale is "better" | `higher_is_better` | `lower_is_better` — tests the single riskiest assumption |
| `weights` | per-dimension weights (only for `weighted_mean`) | `(1,1,1)` | e.g. `(3,1,1)` to weight coverage highest |

## Worked example: from three scores to one sigma

Every observation carries a triple `(coverage, frequency, spatial boundary)`.
With the baseline mapping (`scale_max: 4`, `sigma_rel_best: 0.05`,
`sigma_rel_worst: 1.0`, `combiner: geometric_mean`,
`direction: higher_is_better`) each dimension is first mapped to its own
relative sigma by linear interpolation:

```
frac    = (score − 1) / (scale_max − 1)
frac    = 1 − frac                        # higher_is_better: 4 is the best score
penalty = sigma_rel_best + frac × (sigma_rel_worst − sigma_rel_best)
```

which for a 1–4 scale gives:

| score | frac | relative sigma |
|---|---|---|
| 4 (best) | 0.000 | 0.0500 |
| 3 | 0.333 | 0.3667 |
| 2 | 0.667 | 0.6833 |
| 1 (worst) | 1.000 | 1.0000 |

The three are then combined by geometric mean and multiplied by the reported
value, `sigma = rel × |value|`, floored at 1e-3 kt/y.

**Every triple that occurs in the 2017 data**, with a real observation:

| triple | per-dimension | geometric mean | example value | sigma | as ± |
|---|---|---|---|---|---|
| (4, 3, 4) | 0.0500, 0.3667, 0.0500 | 0.0971 | 4,350.0 | 422.6 | ±9.7% |
| (1, 2, 4) | 1.0000, 0.6833, 0.0500 | 0.3245 | 229.0 | 74.3 | ±32.4% |
| (1, 1, 4) | 1.0000, 1.0000, 0.0500 | 0.3684 | 4,321.0 | 1,591.9 | ±36.8% |
| (1, 1, 3) | 1.0000, 1.0000, 0.3667 | 0.7157 | 506.0 | 362.2 | ±71.6% |
| (1, 1, 1) | 1.0000, 1.0000, 1.0000 | 1.0000 | 3,177.0 | 3,177.0 | ±100.0% |

A triple with no scores at all falls back to
`observation_sigma.observed_rel_sigma` (default 0.10).

Note the spread: the mapping is not a mild adjustment. A well-scored flow is
trusted to ±10% while a poorly-scored one is effectively unconstrained at
±100%, which is why the sweep matters and why the direction of the scale had to
be settled before anything else.

### Sigma is relative, so it is a half-width, not a 95% interval

`r(q)` is applied directly as the likelihood standard deviation:
`sigma = r × |value|`. A flow of 1,000 kt/y scored (4,3,4) therefore carries
`sigma = 97.1` kt/y, and its 95% interval is roughly `1,000 ± 190`, not
`1,000 ± 97`. Any manuscript text describing `r(q)` as defining the interval
itself is off by the ~1.96 factor and should say "standard deviation".

## Direction of the scale

**`higher_is_better` — 4 is the highest confidence.**

Two independent lines of evidence support this.

**Which observations carry which scores.** Under `higher_is_better` the triple
(4,3,4) sits on the large, well-measured USGS quantities — it carries the
4,350 kt/y alumina figure at ±9.7% — while (1,1,1) sits on a lone literature
estimate, at ±100%. Under `lower_is_better` that ordering inverts: the USGS
figure would be trusted to ±90% and the single literature estimate to ±5%.

**Internal consistency of the fit.** Running the inverted mapping
(`scripts/run_pedigree_sensitivity.py`, case
`direction_flipped_lower_is_better`) produces the narrowest credible intervals
of any case in the sweep *and* the worst mass-balance closure — the largest node
residual nearly doubles, from 107 to 201 kt/y. Trusting the poorly-evidenced
values makes the data less self-consistent, not more, which is what an inverted
scale should do.

## What the mapping controls

The sweep separates two things the mapping does. It barely moves the reconciled
values — the largest flows shift by well under a percent across every case — but
it is the dominant control on the **reported uncertainty**:

| mapping | median relative CI width, flows ≥100 kt/y | max node residual (kt/y) |
|---|---|---|
| `direction_flipped_lower_is_better` | 0.34 | 201 |
| `tighter_bounds_best0.02_worst0.5` | 0.41 | 184 |
| `multiplier_0.5x` | 0.41 | 182 |
| **baseline** | **0.48** | **107** |
| `combiner_arithmetic_mean` | 0.66 | 75 |
| `multiplier_2x` | 0.80 | 56 |
| `combiner_max_dimension` | 1.25 | 40 |
| `multiplier_4x` | 1.30 | 32 |

Reported interval widths vary by nearly a factor of four across defensible
mappings, and they trade off directly against closure: tighter observation
sigmas buy narrower intervals at the cost of a worse mass balance. Any statement
about the *precision* of these results should be read against this table, and
`runs/pedigree_sensitivity/indicators.csv` carries the aggregate indicators for
every case.

## Step-by-step review workflow
1. **Manual review** — open `data/raw/aluminum/US aluminum flows.xlsx`,
   sheet `"2017 data"`, and inspect the 6 confidence columns
   (`Confidence - coverage/frequency/spatial boundary (USGS value 1/2)`).
   Check the actual range of values present (expected small integers) and
   look for any other sheet/tab in the workbook that might be an
   undiscovered legend. If USGS methodology documentation for this pedigree
   scheme exists outside this workbook, that is the ground truth to
   reconcile against — update `pedigree_to_rel_sigma`'s docstring and
   defaults once found.
2. **Baseline run** — run the 2017 pipeline once with the current default
   mapping (see command (a) below) and note the MAP mass-balance residual
   and posterior interval widths as your reference point.
3. **Run the sensitivity suite** (command (d) below) — this sweeps
   `configs/pedigree_mapping.yaml`'s case list and produces
   `runs/pedigree_sensitivity/comparison.csv` with, per case: max absolute
   node-balance residual, mean 95% posterior interval width, and posterior
   medians for a chosen set of "key" flows.
4. **Interpret** —
   - If `direction_flipped_higher_is_better` changes key-flow medians
     substantially, the direction assumption is high-stakes and must be
     resolved before publishing results (this is the most important case to
     check first).
   - If `combiner_max_dimension` (worst-dimension-dominates) vs. the default
     geometric mean disagree a lot, that tells you how sensitive results are
     to a single poorly-scored pedigree dimension.
   - `multiplier_0.5x` / `multiplier_2x` / `multiplier_4x` show how much
     interval widths simply scale with overall confidence in the mapping —
     useful for a tornado-style sensitivity plot.
5. **Document the resolution** — once you settle on a mapping (confirmed
   from a codebook, or a defensible expert judgment), update the `baseline`
   block in `configs/pedigree_mapping.yaml` and the
   `observation_sigma.pedigree_mapping` block in `configs/aluminum_2017.yaml`
   to match, and note the resolution in this file's changelog.

## Exact commands to run
See the "Which code to run" section of the top-level `README.md` for the
full set of copy-paste commands (baseline 2017, prior transfer, 2022, and
this sensitivity suite together).

```bash
python scripts/run_pedigree_sensitivity.py \
    --sweep-config configs/pedigree_mapping.yaml \
    --base-config configs/aluminum_2017.yaml \
    --key-flows 0 1 2 3 4
```
Add `--key-flows` values for whichever `flow_idx` positions matter most for
your paper's reported numbers (e.g. the largest flows or ones cited in the
manuscript). Outputs land in `runs/pedigree_sensitivity/`:
- `comparison.csv` — the main table to review
- `_config_<case>.yaml` — the exact resolved config used for each case (for
  reproducibility / debugging)
- `<case>/` — full per-case run outputs (posterior mean CSV, samples, node
  diagnostics), identical structure to a normal pipeline run

## Status

No codebook for the three confidence columns is available, so the mapping is
reconstructed rather than read from a legend.

The scale direction is established by the argument above. The combination rule
and the endpoints are not: `geometric_mean` with 0.05/1.00 endpoints follows
standard pedigree-matrix practice, but nothing in the source materials confirms
it, and it remains the largest modelling assumption in the pipeline.

Every knob is exposed in `configs/pedigree_mapping.yaml`. The sweep should be
reported alongside the headline results so a reader can see how much of the
conclusion rests on that assumption.
