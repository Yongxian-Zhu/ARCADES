"""
arcade_mfa_aluminum.aggregates
------------------------------
System-level quantities derived from the reconciled flows.

Conclusions are stated at the level of the system -- domestic scrap retention,
secondary processing capacity, semi-fabrication scale -- not at the level of
individual flows. Those quantities therefore need one definition used
everywhere: by the robustness sweep that asks whether a conclusion survives an
assumption change, by the comparison against published accounts, and by the
manuscript itself. Defining them here keeps those three consistent.

Each aggregate is a sum over a named set of flows, evaluated on posterior draws
so that every reported total carries a credible interval rather than a point
value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _norm(s) -> str:
    """Canonical node-name form.

    The pipeline stores normalized node names, so both sides of a comparison
    must be normalized the same way or every match silently returns nothing.
    """
    from arcade_mfa_aluminum.adapters.aluminum.adapter import normalize_node_name
    return normalize_node_name(s)


def _match(df: pd.DataFrame, frm=None, to=None,
           frm_in=None, to_in=None, to_prefix=None) -> np.ndarray:
    m = pd.Series(True, index=df.index)
    if frm is not None:
        m &= df["from_node_name"].map(_norm) == _norm(frm)
    if to is not None:
        m &= df["to_node_name"].map(_norm) == _norm(to)
    if frm_in is not None:
        m &= df["from_node_name"].map(_norm).isin([_norm(x) for x in frm_in])
    if to_in is not None:
        m &= df["to_node_name"].map(_norm).isin([_norm(x) for x in to_in])
    if to_prefix is not None:
        m &= df["to_node_name"].map(_norm).str.startswith(_norm(to_prefix))
    return df.loc[m, "flow_idx"].to_numpy(dtype=int)


WROUGHT_ALLOYS = ["1xxx", "2xxx", "3xxx", "4xxx", "5xxx", "6xxx", "7xxx", "8xxx"]
CAST_ALLOYS = ["36x", "38x", "2xx", "3xx", "4xx", "5xx", "7xx", "Other"]


def aggregate_definitions(df: pd.DataFrame) -> dict:
    """Map each named quantity to the flow indices it sums over.

    Names follow the categories used in published U.S. aluminum flow accounts so
    that results are directly comparable.
    """
    return {
        "primary_metal": _match(df, frm="Primary smelting (Hall-Hauroult to aluminum)",
                                to_in=["Remelting", "Refining"]),
        "remelted_metal": _match(df, frm="Remelting", to="Wrought aluminum ingot"),
        "recycled_metal": _match(df, frm="Refining", to="Cast aluminum ingot"),
        "wrought_alloys": _match(df, frm="Wrought aluminum ingot", to_in=WROUGHT_ALLOYS),
        "foundry_alloys": _match(df, frm="Cast aluminum ingot", to_in=CAST_ALLOYS),
        "semis_ingot_total": np.concatenate([
            _match(df, frm="Wrought aluminum ingot", to_in=WROUGHT_ALLOYS),
            _match(df, frm="Cast aluminum ingot", to_in=CAST_ALLOYS),
        ]),
        "sheets_and_foils": np.concatenate([
            _match(df, frm="Rolling", to="Sheet & Plate"),
            _match(df, frm="Rolling", to="Foil"),
        ]),
        "extrusions": _match(df, frm="Extrusion", to="Extruded Products"),
        "cables_and_wires": _match(df, frm="Drawing", to="Wire"),
        "eol_scrap_generated_total": _match(df, to="EOL scrap (generated)"),
        "eol_scrap_recycled": _match(df, frm="EOL scrap (generated)", to="EOL scrap (recycled)"),
        "eol_scrap_exported": _match(df, to="Export EOL scrap"),
        "eol_scrap_lost": _match(df, to="Loss EOL scrap"),
        "scrap_imported": _match(df, frm="Import scrap"),
        "domestic_consumption": _match(df, to_prefix="consumption"),
    }


def derived_ratios(totals: dict) -> dict:
    """Circularity indicators formed from the aggregates.

    Two denominators are in play and they must not be confused:

    * **generated** -- all end-of-life scrap arising in the year, including the
      fraction that is never collected.
    * **collected** -- the part that is actually recovered, i.e. domestically
      recycled plus exported. Uncollected and dissipated material is excluded.

    `eol_collection_rate` is collected / generated. `domestic_retention_rate` is
    the share of *collected* scrap that is recycled domestically rather than
    exported -- the convention used when reporting national circularity, and the
    one the retention argument rests on. Using generated as the denominator
    instead silently folds collection losses into the retention figure and
    understates it substantially.
    """
    out = {}
    gen = totals.get("eol_scrap_generated_total")
    rec = totals.get("eol_scrap_recycled")
    exp = totals.get("eol_scrap_exported")

    if rec is not None and exp is not None:
        collected = rec + exp
        out["eol_scrap_collected_total"] = collected
        out["domestic_retention_rate"] = rec / np.maximum(collected, 1e-9)
        if gen is not None:
            out["eol_collection_rate"] = collected / np.maximum(gen, 1e-9)
            out["eol_uncollected_rate"] = 1.0 - collected / np.maximum(gen, 1e-9)
    if "remelted_metal" in totals and "recycled_metal" in totals:
        out["secondary_metal_total"] = totals["remelted_metal"] + totals["recycled_metal"]
    return out


def compute_aggregates(samples: np.ndarray, df: pd.DataFrame,
                       ci_level: float = 0.95) -> pd.DataFrame:
    """Posterior median and credible interval for every named aggregate.

    `samples` is (n_draws, n_flows). Aggregates are evaluated per draw before
    summarizing, so the interval reflects correlations between flows rather than
    summing independent intervals.
    """
    defs = aggregate_definitions(df)
    totals = {name: samples[:, idx].sum(axis=1) for name, idx in defs.items() if len(idx)}
    totals.update(derived_ratios(totals))

    alpha = 1.0 - ci_level
    rows = []
    for name, draws in totals.items():
        lo, med, hi = np.percentile(draws, [alpha / 2 * 100, 50, (1 - alpha / 2) * 100])
        rows.append({
            "quantity": name,
            "n_flows": len(defs.get(name, [])),
            "median": med,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_width": hi - lo,
            "relative_ci_width": (hi - lo) / max(abs(med), 1e-9),
        })
    return pd.DataFrame(rows)
