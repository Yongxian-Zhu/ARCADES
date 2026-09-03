"""
arcade_mfa_aluminum.plots.sankey_ci
-----------------------------------
Credible-interval Sankey diagrams for the reconciled aluminum flows.

Each link is drawn at the posterior median flow and colored by the width of its
credible interval, in one of two modes: absolute (interval width in flow units)
or relative (width divided by the median, dimensionless). Two static figures
therefore convey both the flow magnitudes and their uncertainty, which is what a
manuscript figure needs.

This follows the uncertainty-on-a-Sankey idea from Lupton & Allwood (2017), but
computes posterior quantiles once, vectorized across all draws, and encodes the
result as a static link color rather than animating over draws. Rendering uses
Plotly, which supports static PDF/SVG export via `kaleido`.

Canonical inputs (produced by `arcade_mfa_aluminum.pipeline.run`):

  <run_dir>/posterior_mean.csv
      Canonical flow schema plus map_estimate / posterior_mean / lower_bound /
      upper_bound. NOTE: lower_bound and upper_bound are FEASIBILITY bounds from
      `pipeline._build_bounds`, not credible intervals -- do not plot them as
      uncertainty.

  <run_dir>/posterior_samples.npy
      (n_draws, n_flows) posterior draws. The sole source of the credible
      intervals computed here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.paths import prepare_output


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class PosteriorFlowSummary:
    """Canonical per-flow table ready for Sankey plotting.

    `df` columns: flow_idx, from_node, to_node, from_label, to_label,
    q_lo, median, q_hi, abs_ci, rel_ci.
    """
    df: pd.DataFrame
    ci_level: float          # e.g. 0.95
    units: str                # e.g. "kt/y"


def _node_label_map(node_catalog: Optional[pd.DataFrame]) -> dict:
    if node_catalog is None or node_catalog.empty:
        return {}
    cols = {c.lower(): c for c in node_catalog.columns}
    num_col = cols.get("node_number")
    name_col = cols.get("node_name")
    if num_col is None or name_col is None:
        return {}
    return dict(zip(node_catalog[num_col], node_catalog[name_col]))


def load_node_catalog(workbook_path: str, sheet_name: str = "node_catalog") -> pd.DataFrame:
    """Load the node_catalog sheet (node_number, node_name, node_description)."""
    try:
        raw = pd.read_excel(workbook_path, sheet_name=sheet_name)
        raw.columns = [str(c).strip() for c in raw.columns]
        return raw
    except Exception as e:
        warnings.warn(f"Could not load node_catalog from {workbook_path}: {e}")
        return pd.DataFrame()


def load_flow_catalog(workbook_path: str, sheet_name: str = "flow_catalog") -> pd.DataFrame:
    """Load the flow_catalog sheet (flow_index, flow_description)."""
    try:
        raw = pd.read_excel(workbook_path, sheet_name=sheet_name)
        raw.columns = [str(c).strip() for c in raw.columns]
        return raw
    except Exception as e:
        warnings.warn(f"Could not load flow_catalog from {workbook_path}: {e}")
        return pd.DataFrame()


def build_posterior_flow_summary(
    posterior_mean_csv_path: str,
    posterior_samples_npy_path: str,
    *,
    ci_level: float = 0.95,
    workbook_path: Optional[str] = None,
    node_catalog_sheet: str = "node_catalog",
    units: str = "kt/y",
    unit_scale: float = 1.0,
    min_median_for_rel_ci: float = 1e-6,
) -> PosteriorFlowSummary:
    """Build the canonical per-flow summary table used by both Sankey figures.

    Parameters
    ----------
    posterior_mean_csv_path : path to `posterior_mean.csv` written
        by `pipeline.run()` -- supplies flow_idx/from_node/to_node/
        from_node_name/to_node_name (canonical schema).
    posterior_samples_npy_path : path to `posterior_samples.npy`
        (n_draws, n_flows) written by `pipeline.run()` -- the actual
        posterior draws used to compute credible-interval quantiles.
    ci_level : central credible interval width, e.g. 0.95 for a 95% CI
        (quantiles taken at (1-ci_level)/2 and 1-(1-ci_level)/2).
    workbook_path : if given, the node_catalog sheet is loaded and used to
        prefer clean node_name labels over the normalized from/to_node_name
        already in the CSV (falls back silently if unavailable).
    unit_scale : multiply raw flow values by this factor for display, e.g.
        0.001 to convert kt/y -> Mt/y. Applied to median/q_lo/q_hi/abs_ci
        (rel_ci is dimensionless and unaffected by the scale choice).
    min_median_for_rel_ci : flows with |median| below this floor (in
        *original*, pre-scale units) get rel_ci = NaN to avoid a
        division-by-near-zero blowup dominating the relative-CI color scale.
    """
    df = pd.read_csv(posterior_mean_csv_path)
    samples = np.load(posterior_samples_npy_path)   # (n_draws, n_flows)

    if samples.shape[1] != len(df):
        raise ValueError(
            f"Posterior samples n_flows={samples.shape[1]} does not match "
            f"posterior_mean_csv rows={len(df)}. Are these from the same run?"
        )

    alpha = 1.0 - ci_level
    median_raw = np.median(samples, axis=0)
    q_lo = np.quantile(samples, alpha / 2.0, axis=0) * unit_scale
    q_hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0) * unit_scale
    median = median_raw * unit_scale

    abs_ci = q_hi - q_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_ci = np.where(np.abs(median_raw) >= min_median_for_rel_ci,
                           abs_ci / np.abs(median), np.nan)

    out = pd.DataFrame({
        "flow_idx": df["flow_idx"].to_numpy(),
        "from_node": df["from_node"].to_numpy(),
        "to_node": df["to_node"].to_numpy(),
        "from_label": df["from_node_name"].astype(str),
        "to_label": df["to_node_name"].astype(str),
        "q_lo": q_lo,
        "median": median,
        "q_hi": q_hi,
        "abs_ci": abs_ci,
        "rel_ci": rel_ci,
    })

    if workbook_path:
        catalog = load_node_catalog(workbook_path, node_catalog_sheet)
        label_map = _node_label_map(catalog)
        if label_map:
            out["from_label"] = out["from_node"].map(label_map).fillna(out["from_label"])
            out["to_label"] = out["to_node"].map(label_map).fillna(out["to_label"])

    return PosteriorFlowSummary(df=out, ci_level=ci_level, units=units)


# ---------------------------------------------------------------------------
# Sankey construction
# ---------------------------------------------------------------------------

def _hex_colorscale(values: np.ndarray, colorscale: str = "YlOrRd") -> list:
    """Map an array of values to hex colors via a Plotly named colorscale.
    NaN values (e.g. undefined rel_ci for near-zero flows) render grey."""
    import plotly.colors as pc

    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return ["#888888"] * len(vals)
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    scale = pc.get_colorscale(colorscale)
    colors = []
    for v in vals:
        if not np.isfinite(v):
            colors.append("#cccccc")
            continue
        t = float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))
        colors.append(pc.sample_colorscale(scale, [t])[0])
    return colors


def build_sankey_frame(
    summary: PosteriorFlowSummary,
    *,
    mode: Literal["absolute", "relative"] = "absolute",
    min_flow: float = 0.0,
    node_order: Optional[list] = None,
    colorscale: str = "YlOrRd",
) -> dict:
    """Turn a PosteriorFlowSummary into Plotly go.Sankey-ready node/link dicts.

    `mode="absolute"` colors links by abs_ci (q_hi - q_lo, same units as the
    flow value). `mode="relative"` colors links by rel_ci (dimensionless
    CI width / |median|), with NaN (near-zero-median flows) rendered grey.

    Link width is always the posterior median flow (x_pq in the flow-based
    formulation); see module docstring for the optional node-allocation
    (w_pq) coloring path via `allocation_share_ci`.
    """
    df = summary.df.copy()
    df = df[df["median"].abs() >= min_flow].reset_index(drop=True)

    color_field = "abs_ci" if mode == "absolute" else "rel_ci"

    # Deterministic node ordering: explicit node_order if given, else order
    # of first appearance (from_label then to_label per row), matching the
    # upstream->downstream reading order typical of MFA Sankeys and giving
    # reproducible layouts across repeated runs of the same config.
    if node_order is not None:
        labels_in_order = list(node_order)
    else:
        seen = []
        for col in ("from_label", "to_label"):
            for v in df[col]:
                if v not in seen:
                    seen.append(v)
        labels_in_order = seen

    label_to_idx = {lab: i for i, lab in enumerate(labels_in_order)}
    df = df[df["from_label"].isin(label_to_idx) & df["to_label"].isin(label_to_idx)]

    link_colors = _hex_colorscale(df[color_field].to_numpy(), colorscale=colorscale)

    hover = []
    for row in df.itertuples():
        rel_txt = f"{row.rel_ci:.2f}x" if np.isfinite(row.rel_ci) else "n/a (~0 flow)"
        hover.append(
            f"{row.from_label} -> {row.to_label}<br>"
            f"median: {row.median:,.2f} {summary.units}<br>"
            f"{int(summary.ci_level * 100)}% CI: [{row.q_lo:,.2f}, {row.q_hi:,.2f}] {summary.units}<br>"
            f"abs CI width: {row.abs_ci:,.2f} {summary.units}<br>"
            f"rel CI width: {rel_txt}"
        )

    return {
        "node": {"label": labels_in_order, "pad": 12, "thickness": 14},
        "link": {
            "source": df["from_label"].map(label_to_idx).to_list(),
            "target": df["to_label"].map(label_to_idx).to_list(),
            "value": df["median"].clip(lower=0).to_list(),
            "color": link_colors,
            "customdata": hover,
            "hovertemplate": "%{customdata}<extra></extra>",
        },
        "color_field": color_field,
        "color_values": df[color_field].to_list(),
    }


def make_sankey_figure(
    summary: PosteriorFlowSummary,
    *,
    mode: Literal["absolute", "relative"] = "absolute",
    title: Optional[str] = None,
    min_flow: float = 0.0,
    node_order: Optional[list] = None,
    colorscale: str = "YlOrRd",
):
    """Build a Plotly Figure for one CI Sankey (absolute or relative)."""
    import plotly.graph_objects as go

    frame = build_sankey_frame(
        summary, mode=mode, min_flow=min_flow, node_order=node_order, colorscale=colorscale
    )
    fig = go.Figure(go.Sankey(node=frame["node"], link=frame["link"]))

    default_title = (
        f"Aluminum MFA flows -- link color = {int(summary.ci_level * 100)}% "
        f"credible interval width "
        f"({'absolute, ' + summary.units if mode == 'absolute' else 'relative, dimensionless'})"
    )
    fig.update_layout(
        title=title or default_title,
        font=dict(size=11),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_figure(fig, out_prefix: str, *, formats=("pdf", "svg", "png", "html")) -> dict:
    """Save a Plotly figure to multiple formats. PDF/SVG/PNG require the
    `kaleido` package (static image export); HTML always works standalone.

    Returns dict of format -> path actually written; formats that fail
    (e.g. kaleido not installed) are skipped with a warning, not an error.
    """
    written = {}
    for fmt in formats:
        path = f"{out_prefix}.{fmt}"
        try:
            target = prepare_output(path)
            if fmt == "html":
                fig.write_html(target, include_plotlyjs="cdn")
            else:
                fig.write_image(target, scale=2 if fmt == "png" else 1)
            written[fmt] = path
        except Exception as e:
            warnings.warn(f"Could not export {fmt.upper()} to {path}: {e}. "
                           f"(Static PDF/SVG/PNG export requires `pip install -U kaleido`.)")
    return written
