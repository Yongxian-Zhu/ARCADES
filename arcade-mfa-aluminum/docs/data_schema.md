# Data schema reference

## Source: `data/raw/aluminum/US aluminum flows.xlsx`

234 flow rows, 105 nodes, four sheets. This workbook is the single authoritative
input; no other ingestion path exists.

### Sheet "2017 data" (234 rows) — 2017 input

| Column | Type | Notes |
|---|---|---|
| `flow_index` | int, 1..234 | 1-indexed; converted to 0-indexed `flow_idx` on ingestion |
| `data_year` | int | 2017 |
| `from_node_number` / `to_node_number` | int | node ids, matching `node_catalog` |
| `from_node_name` / `to_node_name` / `*_description` | str | human-readable labels |
| `flow_description` | str | |
| `USGS value 1` | float | observed flow value |
| `Confidence - coverage / frequency / spatial boundary (USGS value 1)` | float, ~1–4 | pedigree-matrix confidence dimensions; the scale direction and combination rule are an **assumed** interpretation (see `pedigree_to_rel_sigma`) |
| `Notes (USGS value 1)` | str | free text |
| `USGS value 2` + its three confidence dimensions + notes | | second source; **used in the likelihood**, not a passive cross-check |
| `Other literature` + its three confidence dimensions + notes | float | third source; **used in the likelihood**. 13 values covering process-level routing (Van den Eynde et al. 2022) and mill-to-transport shipments (Hua et al. 2022) |

### Sheet "2022 results" (234 rows)

| Column | Notes |
|---|---|
| `flow_index`, `result_year`, `from`/`to_node_*`, `flow_description` | as above |
| `observation_2022` | the 2022 observation. Populated for 49 of 234 flows; the rest are unobserved |
| `posterior_mean_2022`, `lower_bound_2022`, `upper_bound_2022`, `posterior_std_2022`, `prior_mean_2022` | outputs of an earlier reconciliation, retained as reference columns for comparison. **Not** fed back in as inputs |
| `has_observation_2022` | bool |

### Sheet "node_catalog" (105 rows)

`node_number, node_name, node_description` — canonical node list. Node numbers
are stable identifiers and are **not** contiguous: ids 11, 13, 14 and 15 are
absent (see below).

### Sheet "flow_catalog" (234 rows)

`flow_index, flow_description`.

## Network conventions

Two points about the network that are easy to get wrong when reading the
workbook.

**Node numbering is not contiguous.** `node_number` is a stable identifier;
gaps are expected. Index by `flow_idx` (0-based, contiguous) rather than by node
number.

**Imported unwrought ingot is remelted before casting.** Flows 14 and 15 are a
matched pair -- 14 to Remelting (the wrought route) and 15 to Refining (the cast
route) -- so imported metal passes through a melting step rather than entering
an ingot pool directly. Wrought aluminum ingot therefore has a single inflow,
from Remelting, and its mass balance ties that flow to the sum of the eight
alloy-class outflows plus wrought-ingot exports.

## Source precedence and how conflicts are resolved

There is **no precedence order**. Every reported value enters the likelihood as
its own term:

```
0.5 Σ_j Σ_s (x_j − y_sj)² / σ_sj²
```

so a flow with two sources is pulled toward both, in inverse proportion to their
variances, and settles between them nearer the tighter one. Earlier versions read
only `USGS value 1`, which meant the first column silently won every
disagreement; 29 flows in 2017 carry more than one observation and those
conflicts are now resolved on stated uncertainty rather than column order.

Each source derives its sigma from its own pedigree triple
(`docs/pedigree_sensitivity.md`), adjustable per source via
`observation_sources.<name>.sigma_multiplier` and per flow via
`observation_overrides`. Every observation that entered a fit, with its source,
scores and derived sigma, is written to `source_inventory.csv`.

### Aggregate sources

Sources publishing at coarser resolution than the model constrain a **sum**
rather than any single flow (`aggregate_observations`, see
`configs/aluminum_association_2022.yaml`). Splitting a reported total across
flows in proportion to another source would invent detail the source never
published and partly restate the source being compared against. Reported against
reconciled sums in `aggregate_observations.csv`.

### The USGS / Aluminum Association discrepancy (R3-2)

For 2022 imports of unwrought ingot the two sources disagree materially:

| source | value | how it enters |
|---|---|---|
| USGS | 4,150 kt/y | two per-flow observations (2,829.8 to Remelting, 1,320.2 to Refining), 5% customs sigma |
| Aluminum Association | 3,649 kt/y | one aggregate constraint on their sum, 5% |

Both are retained. The reconciliation lands between them, and the residual is
reported in `aggregate_observations.csv` so the disagreement stays visible rather
than being resolved by fiat.

**Recovery of Secondary is deliberately excluded** from the AA constraints. AA
excludes internal run-around scrap, which this network cannot separate from
traded scrap — the flows it captures include both — so the two quantities are not
commensurable. It is used as a validation check instead. Apparent Consumption,
Inventory Change and Aluminum Consumption are likewise excluded as aggregate
indicators on a different system boundary.

### Reported zeros

A reported `0.0` is treated as "not measured" rather than "provably absent": the
lower bound stays at zero and the upper bound is widened by
`bounds.zero_flow_upper` (default 1.0 kt/y). Pinning such a flow to a negligible
interval would assert more certainty than the data support and would stop mass
balance routing anything through it. Twelve flows carry an exact `0.0` in the
2017 data.

Two consequences matter when reading results. A 2017 zero propagates into the
2022 transfer prior, so a flow reported as zero in 2017 but non-zero in 2022 can
be held near zero by the prior and its allocation share — this is the mechanism
behind the largest remaining 2022 observation residuals. And a zero does not
prevent mass balance assigning real flow through that route.

## Canonical in-memory schema

`arcade_mfa_aluminum.adapters.aluminum.adapter.CANONICAL_FLOW_COLUMNS`:

```
flow_idx (0-indexed int), from_node, to_node, from_node_name, to_node_name,
value_1, value_2, value_3, value_4, year
```

Both year adapters normalize into this single schema, so `graph.py`,
`inference/*` and `pipeline.py` never depend on which sheet was read.

## Derived artifacts

| Path | Contents |
|---|---|
| `data/canonical/aluminum/prior_mean_2017_to_2022.npy` / `prior_cov_*.npy` | 2017 posterior summarized as the 2022 prior |
| `data/canonical/aluminum/allocation_shares_2017.npz` | per-node allocation shares and their posterior sd |

The prior and the allocation shares are addressed by `flow_idx` and are
therefore specific to a given flow set. Both loaders check dimensions and raise
rather than silently misaligning if the flow set has changed.
