# Credible-interval Sankey diagrams

Two reviewer-ready Sankey figures visualize posterior flow uncertainty:

1. **Absolute CI Sankey** — link color = `q_hi - q_lo` (e.g. q97.5 - q2.5)
   in display units (kt/y or Mt/y). Highlights which flows have the widest
   *absolute* uncertainty band — typically the largest flows in the network,
   since absolute uncertainty tends to scale with flow magnitude.
2. **Relative CI Sankey** — link color = `(q_hi - q_lo) / |median|`,
   dimensionless. Highlights which flows are *proportionally* least well
   constrained by the data/model, regardless of their size — often small
   flows with sparse observational support (e.g. minor scrap export
   pathways) rather than the dominant flows.

Both figures use the **same link widths** (posterior median flow, `q50`),
so the two are visually comparable side-by-side: same network skeleton,
different color encoding of uncertainty.

## Data source (what gets read)

Both figures read directly from a completed `pipeline.run()` output
directory (`runs/<run_name>/`):

| File | Role |
|---|---|
| `posterior_mean.csv` | canonical flow schema: `flow_idx`, `from_node`, `to_node`, `from_node_name`, `to_node_name` (node topology + labels) |
| `posterior_samples.npy` | `(n_draws, n_flows)` posterior draws — **the sole source of the credible-interval quantiles** |

**Important**: `posterior_mean.csv` also contains `lower_bound`/`upper_bound`
columns, but those are *feasibility bounds* used to constrain the MAP solve
(0.2x/1.8x the observed value in `pipeline._build_bounds`), **not** credible
interval quantiles. The plotting code does not use them — it recomputes
`q_lo`/`q_hi` directly from `posterior_samples.npy` via `np.quantile`, so the
CI level (90%, 95%, ...) is under your control at plot time, not fixed at
run time.

If your run only saved a summary table with quantile columns already
computed (no raw draws saved), you have two options:
- Re-run with the samples saved (default behavior of `pipeline.run()`), or
- Adapt `build_posterior_flow_summary` to read `q_lo`/`q_hi` columns
  directly from your summary CSV instead of computing them from
  `posterior_samples.npy` — the rest of the pipeline (Sankey construction,
  coloring, export) is agnostic to how the summary table was produced.

## Running it

```bash
# 1. Run the pipeline first (produces the two canonical files above)
python -m arcade_mfa_aluminum.cli run --config configs/aluminum_2017.yaml

# 2. Generate both CI Sankeys (absolute + relative) for that run
python scripts/plot_sankey_ci.py --run runs/<2017_run_id>

# Or just one:
python scripts/plot_sankey_ci.py --run runs/<2017_run_id> --mode absolute
python scripts/plot_sankey_ci.py --run runs/<2022_run_id> --mode relative

# Display in Mt/y instead of raw kt/y, use a 90% CI, drop tiny flows,
# and pull clean node labels from the workbook's node_catalog sheet:
python scripts/plot_sankey_ci.py \
    --run runs/<2017_run_id> \
    --units "Mt/y" --unit-scale 0.001 \
    --ci-level 0.90 \
    --min-flow 0.5 \
    --workbook "data/raw/aluminum/US aluminum flows.xlsx" \
    --colorscale Viridis
```

Outputs land in `<run>/plots/` by default (override with `--out`):
- `sankey_absolute_ci.pdf` / `.svg` / `.png` / `.html`
- `sankey_relative_ci.pdf` / `.svg` / `.png` / `.html`

PDF/SVG/PNG require `kaleido` (`pip install -U kaleido`, already listed in
`requirements.txt`/`pyproject.toml`); the interactive HTML export always
works standalone (embeds Plotly via CDN) and is useful for exploring
individual flow tooltips (hover shows median, CI bounds, and both
uncertainty measures per flow) before committing to a static figure.

## Configuration knobs

| Flag | Default | Effect |
|---|---|---|
| `--ci-level` | `0.95` | Central credible interval width (e.g. `0.90` for a 90% CI) |
| `--units` / `--unit-scale` | `kt/y` / `1.0` | Display unit label + multiplicative conversion (raw workbook values are kt/y) |
| `--min-flow` | `0.0` | Drop flows with `|median|` below this threshold — decluttering for dense networks |
| `--colorscale` | `YlOrRd` | Any [Plotly named colorscale](https://plotly.com/python/builtin-colorscales/) |
| `--workbook` | none | If given, node labels are pulled from the workbook's `node_catalog` sheet (clean names) instead of the normalized `from/to_node_name` already in the CSV |
| `--formats` | `pdf svg png html` | Which export formats to attempt |

Node ordering in the current implementation follows first-appearance order
in the (filtered) flow table — deterministic across repeated runs of the
same config/filters, but not yet a hand-curated upstream→downstream layout.
Pass an explicit `node_order` list to `build_sankey_frame()` directly (not
yet exposed as a CLI flag) if you need manuscript-specific node positioning.

## Adaptation from prior literature (Lupton & Allwood / bayesian-mfa-mcmc)

The reference implementation in `bayesian-mfa-mcmc/trace_sankey_helpers.py`
(Lupton & Allwood 2017, steel MFA with node-allocation processes) builds one
Sankey per MCMC sample or the posterior mean using `sankeyview` /
`ipysankeywidget`, and *animates* over draws (`animate_samples`) to convey
uncertainty visually in a Jupyter widget.

This package instead:
- Computes posterior quantiles **once**, vectorized over all draws
  (`np.quantile` on the `(n_draws, n_flows)` array), and encodes the
  resulting CI width as a **static** link color — one absolute-CI figure and
  one relative-CI figure, rather than an animation loop. This is what a
  reviewer/manuscript figure pipeline needs (exportable PDF/SVG), whereas
  the interactive animation is better suited to live exploration.
- Uses Plotly (`go.Sankey` + `kaleido`) instead of
  `sankeyview`/`ipysankeywidget`, since the latter is Jupyter-widget-only
  with no static vector export path.
- Sources node/flow topology from this package's canonical
  `AluminumFlowTable` schema (`flow_idx`/`from_node`/`to_node`/
  `from_node_name`/`to_node_name`) rather than a raw process-adjacency
  trace (`I`, `F` matrices) — the long-dataframe shape
  (source → target → value, one row per flow) is preserved from
  `inputs_flows_as_dataframe`, just re-sourced.

## Limitations

- **Node ordering is not curated for publication.** First-appearance order is
  deterministic but arbitrary. For a manuscript figure, specify `node_order`
  explicitly — for example grouping by life-cycle stage (primary production →
  semis → end use → end of life → recycling). `build_sankey_frame` accepts this
  via `node_order=[...]`.
- **Link width uses the posterior median and clips negatives to zero.** A flow
  whose median is negative would render as a zero-width link while still
  carrying its credible-interval color. With non-negative bounds enforced this
  should not arise, but it is worth checking before publishing a figure.
- **Static export requires `kaleido`.** Without it the HTML output still works
  and PDF/SVG/PNG export is skipped with a warning rather than failing.
