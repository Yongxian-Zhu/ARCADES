"""
arcade_mfa_aluminum.adapters.aluminum.adapter
-------------------------------------
Ingestion adapter for the U.S. aluminum flow network (2017 baseline + 2022
update)

Workbook schema (234 flows, 105 nodes):
  flow_index runs contiguously 1..234. node_number is a stable identifier
  and is NOT contiguous; gaps are expected and must not be relied on for
  ordering or indexing.
  "2017 data"    -- flow_index, data_year, from_node_number, from_node_name,
                     from_node_description, to_node_number, to_node_name,
                     to_node_description, flow_description,
                     "USGS value 1", "Confidence - coverage (USGS value 1)",
                     "Confidence - frequency (USGS value 1)",
                     "Confidence - spatial boundary (USGS value 1)",
                     "Notes (USGS value 1)", "USGS value 2", (...same 4
                     confidence/notes fields for value 2...), "Other literature"
  "node_catalog" -- node_number, node_name, node_description  (104 rows)
  "flow_catalog" -- flow_index, flow_description                (233 rows)

The 2017 sheet carries THREE independent confidence dimensions per value
(coverage / frequency / spatial boundary, each presumably on a small integer
scale e.g. 1-4) `pedigree_to_rel_sigma()` below
combines these into one relative-sigma estimate (pedigree-matrix style,
common in MFA/LCA uncertainty practice: geometric mean of per-dimension
multipliers).

"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd




AUTHORITATIVE_2017_SOURCES = {
    "2017 data",                        # authoritative workbook sheet
    "us aluminum flows - 2017 data",
}

DEPRECATED_2017_SOURCES = {
    "flow_data",
    "flow_data_old",
    "flow_data_draft",
}

PREFERRED_2017_SOURCE = "2017 data"     # now the workbook sheet, not the CSV
AUTHORITATIVE_2022_SOURCES = {"2022 results"}
PREFERRED_2022_SOURCE = "2022 results"

DEFAULT_WORKBOOK_PATH = "data/raw/aluminum/US aluminum flows.xlsx"


class NonAuthoritative2017SourceError(ValueError):
    """Raised when a 2017 load is attempted against a non-authoritative source."""


def _stem(name: str) -> str:
    base = os.path.basename(str(name))
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base.strip().lower()


def assert_authoritative_2017(source_name: str) -> None:
    """Hard gate: raise unless `source_name` is a recognized authoritative 2017 source."""
    key = _stem(source_name)
    if key in {_stem(s) for s in DEPRECATED_2017_SOURCES}:
        raise NonAuthoritative2017SourceError(
            f"'{source_name}' is a DEPRECATED 2017 source. "
            f"Use one of: {sorted(AUTHORITATIVE_2017_SOURCES)}"
        )
    if key not in {_stem(s) for s in AUTHORITATIVE_2017_SOURCES}:
        raise NonAuthoritative2017SourceError(
            f"'{source_name}' is not a recognized authoritative 2017 source. "
            f"Authoritative sources are: {sorted(AUTHORITATIVE_2017_SOURCES)}. "
            f"If this is a legitimate new vintage, add it to AUTHORITATIVE_2017_SOURCES "
            f"in arcade_mfa_aluminum/adapters/aluminum/adapter.py after confirming with the user."
        )


def assert_authoritative_2022(source_name: str) -> None:
    key = _stem(source_name)
    if key not in {_stem(s) for s in AUTHORITATIVE_2022_SOURCES}:
        raise ValueError(
            f"'{source_name}' is not a recognized authoritative 2022 source. "
            f"Authoritative sources: {sorted(AUTHORITATIVE_2022_SOURCES)}"
        )


# =============================================================================
# CANONICAL SCHEMA
# =============================================================================

CANONICAL_FLOW_COLUMNS = [
    "flow_idx", "from_node", "to_node", "from_node_name", "to_node_name",
    "value_1", "value_2", "value_3", "value_4", "year",
]


@dataclass
class AluminumFlowTable:
    year: int
    source_name: str
    df: pd.DataFrame                 # canonical schema (CANONICAL_FLOW_COLUMNS)
    score_df: Optional[pd.DataFrame] = None   # pedigree/confidence, raw column form
    observations: Optional[pd.DataFrame] = None  # long form, one row per observation
    n_flows: int = field(init=False)

    def __post_init__(self):
        self.n_flows = len(self.df)


def normalize_node_name(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^\w_]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


# -----------------------------------------------------------------------
# Workbook (authoritative) loaders
# -----------------------------------------------------------------------

def load_aluminum_2017_from_workbook(
    workbook_path: str = DEFAULT_WORKBOOK_PATH,
    *,
    sheet_name: str = "2017 data",
) -> AluminumFlowTable:
    """Authoritative 2017 loader from the "2017 data" sheet of
    "US aluminum flows.xlsx". Raises NonAuthoritative2017SourceError if
    `sheet_name` is not whitelisted."""
    assert_authoritative_2017(sheet_name)

    raw = pd.read_excel(workbook_path, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]

    required = ["flow_index", "from_node_number", "to_node_number",
                "from_node_name", "to_node_name"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{workbook_path}[{sheet_name}]: missing required columns {missing}")

    for c in ["flow_index", "from_node_number", "to_node_number",
              "USGS value 1", "USGS value 2"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # `flow_index` in the workbook is 1-based (1..233). Convert to a 0-based
    # `flow_idx` here so that it matches
    # canonical array position everywhere downstream (mass-balance matrix
    # columns, posterior_samples.npy columns, etc.) -- mirrors the same
    # "-1" conversion already applied in load_aluminum_2022_from_workbook().
    df = pd.DataFrame({
        "flow_idx": (raw["flow_index"] - 1).astype("Int64"),
        "from_node": raw["from_node_number"].astype("Int64"),
        "to_node": raw["to_node_number"].astype("Int64"),
        "from_node_name": raw["from_node_name"].apply(normalize_node_name),
        "to_node_name": raw["to_node_name"].apply(normalize_node_name),
        "value_1": raw.get("USGS value 1", np.nan),
        "value_2": raw.get("USGS value 2", np.nan),
        "value_3": np.nan,   # the workbook carries only two USGS value columns
        "value_4": np.nan,
        "year": raw.get("data_year", 2017),
    })[CANONICAL_FLOW_COLUMNS]

    if len(df) and (df["flow_idx"].min() != 0 or df["flow_idx"].max() != len(df) - 1):
        raise ValueError(
            f"{workbook_path}[{sheet_name}]: after 0-basing, flow_idx range is "
            f"[{df['flow_idx'].min()}, {df['flow_idx'].max()}] but expected "
            f"[0, {len(df) - 1}] for {len(df)} flows. The workbook's flow_index "
            f"numbering may have changed (gaps, duplicates, or a different base) -- "
            f"re-check the -1 conversion above before trusting downstream results."
        )

    # Pedigree/confidence columns kept in their raw (long) form -- see
    # pedigree_to_rel_sigma() for how these three dimensions combine.
    pedigree_cols = [
        "flow_index",
        "Confidence - coverage (USGS value 1)",
        "Confidence - frequency (USGS value 1)",
        "Confidence - spatial boundary (USGS value 1)",
        "Confidence - coverage (USGS value 2)",
        "Confidence - frequency (USGS value 2)",
        "Confidence - spatial boundary (USGS value 2)",
    ]
    score_df = raw[[c for c in pedigree_cols if c in raw.columns]].rename(
        columns={"flow_index": "flow_idx"}
    )
    # Match the 0-based convention used in `df` above -- score_df is not
    # currently joined by flow_idx value anywhere in pipeline.py (it's
    # consumed by row-position, same row order as `df` since both come
    # from the same `raw` read), but keep it 0-based for consistency in
    # case a future caller joins on this column.
    if "flow_idx" in score_df.columns:
        score_df = score_df.assign(flow_idx=score_df["flow_idx"] - 1)

    return AluminumFlowTable(
        year=2017, source_name=sheet_name, df=df, score_df=score_df,
        observations=extract_observations(raw),
    )


def load_aluminum_2022_from_workbook(
    workbook_path: str = DEFAULT_WORKBOOK_PATH,
    *,
    sheet_name: str = "2022 results",
) -> AluminumFlowTable:
    """Loader for the "2022 results" sheet. NOTE: this sheet already contains
    posterior_mean_2022 / prior_mean_2022 / observation_2022 -- i.e. it looks
    like a PRIOR RUN'S OUTPUT, not raw 2022 observations. Use
    `observation_2022` as the y_obs input for a fresh 2022 MAP run, and treat
    `posterior_mean_2022` / `prior_mean_2022` as reference values for
    validating the new pipeline against the original run that produced this
    sheet (see diagnostics.compare_to_prior_run)."""
    assert_authoritative_2022(sheet_name)

    raw = pd.read_excel(workbook_path, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]

    for c in ["flow_index", "from_node_number", "to_node_number", "observation_2022",
              "posterior_mean_2022", "prior_mean_2022", "lower_bound_2022",
              "upper_bound_2022", "posterior_std_2022"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # See load_aluminum_2017_from_workbook for why flow_index is converted
    # from 1-indexed (source) to 0-indexed (canonical array position) here.
    df = pd.DataFrame({
        "flow_idx": (raw["flow_index"] - 1).astype("Int64"),
        "from_node": raw["from_node_number"].astype("Int64"),
        "to_node": raw["to_node_number"].astype("Int64"),
        "from_node_name": raw["from_node_name"].apply(normalize_node_name),
        "to_node_name": raw["to_node_name"].apply(normalize_node_name),
        "value_1": raw.get("observation_2022", np.nan),   # treat as the "observation" column
        "value_2": np.nan,
        "value_3": np.nan,
        "value_4": np.nan,
        "year": raw.get("result_year", 2022),
    })[CANONICAL_FLOW_COLUMNS]

    reference_cols = ["flow_index", "posterior_mean_2022", "lower_bound_2022",
                       "upper_bound_2022", "posterior_std_2022", "prior_mean_2022",
                       "has_observation_2022"]
    score_df = raw[[c for c in reference_cols if c in raw.columns]].rename(
        columns={"flow_index": "flow_idx"}
    )

    return AluminumFlowTable(
        year=2022, source_name=sheet_name, df=df, score_df=score_df,
        observations=extract_observations(raw),
    )



# -----------------------------------------------------------------------
# Observation extraction (all sources)
# -----------------------------------------------------------------------

#: Value columns that are treated as observations even when they carry no
#: pedigree triple. Any column X for which "Confidence - coverage (X)" exists is
#: discovered automatically, so adding a new source to the workbook needs no
#: code change -- add the value column and its three confidence columns.
OBSERVATION_COLUMNS_WITHOUT_PEDIGREE = ("observation_2022",)


def discover_observation_columns(raw: pd.DataFrame) -> dict:
    """Map each observation column to its (coverage, frequency, spatial) triple.

    A column is an observation column if a matching
    ``Confidence - coverage (<column>)`` exists, or if it is named in
    `OBSERVATION_COLUMNS_WITHOUT_PEDIGREE`. The value is None where no pedigree
    triple accompanies the column.
    """
    found = {}
    for col in raw.columns:
        cov = f"Confidence - coverage ({col})"
        if cov in raw.columns:
            found[col] = (
                cov,
                f"Confidence - frequency ({col})",
                f"Confidence - spatial boundary ({col})",
            )
    for col in OBSERVATION_COLUMNS_WITHOUT_PEDIGREE:
        if col in raw.columns and col not in found:
            found[col] = None
    return found


def extract_observations(raw: pd.DataFrame) -> pd.DataFrame:
    """Every reported value in the sheet, one row per (flow, source).

    Columns: flow_idx (0-based), source, value, coverage, frequency, spatial.

    A flow may carry several observations from different sources; all of them
    are returned. Reconciling them is the model's job, not the loader's -- so
    nothing is dropped here, including values that disagree with each other.
    """
    cols = discover_observation_columns(raw)
    idx0 = pd.to_numeric(raw["flow_index"], errors="coerce") - 1
    frames = []
    for col, ped in cols.items():
        v = pd.to_numeric(raw[col], errors="coerce")
        keep = v.notna() & idx0.notna()
        if not keep.any():
            continue
        rec = pd.DataFrame({
            "flow_idx": idx0[keep].astype(int).to_numpy(),
            "source": col,
            "value": v[keep].astype(float).to_numpy(),
        })
        for name, c in zip(("coverage", "frequency", "spatial"),
                           ped if ped else (None, None, None)):
            rec[name] = (
                pd.to_numeric(raw.loc[keep, c], errors="coerce").to_numpy()
                if c and c in raw.columns else np.nan
            )
        frames.append(rec)
    if not frames:
        return pd.DataFrame(
            columns=["flow_idx", "source", "value", "coverage", "frequency", "spatial"]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["flow_idx", "source"], ignore_index=True
    )

# -----------------------------------------------------------------------
# Pedigree (workbook) -> relative sigma
# -----------------------------------------------------------------------

def pedigree_to_rel_sigma(
    coverage: float, frequency: float, spatial_boundary: float,
    *,
    scale_max: float = 4.0,
    sigma_rel_best: float = 0.05,
    sigma_rel_worst: float = 1.0,
    global_multiplier: float = 1.0,
    combiner: str = "geometric_mean",
    direction: str = "higher_is_better",
    weights: tuple[float, float, float] | None = None,
) -> float:
    """Combine the 3 pedigree-matrix dimensions from the "2017 data" sheet
    (coverage / frequency / spatial boundary, each on an assumed 1-scale_max
    integer scale -- confirmed from sample rows: values of 1, 2, 4 observed)
    into one relative-sigma estimate.

    This is the SINGLE most important assumption-laden function in the
    aluminum pipeline: no codebook for these 3 columns was found in the
    the source materials, so both the scale direction and the combination rule
    are inferred, not confirmed. Every knob here is exposed so a systematic
    sensitivity sweep can be run without touching code -- see
    `docs/pedigree_sensitivity.md` and `scripts/run_pedigree_sensitivity.py`,
    driven by `configs/pedigree_mapping.yaml`.

    Parameters
    ----------
    scale_max : assumed top of the raw pedigree scale (1..scale_max).
    sigma_rel_best / sigma_rel_worst : relative-sigma bounds mapped onto the
        best/worst pedigree score.
    global_multiplier : uniform scale factor applied to the final relative
        sigma -- the simplest, coarsest sensitivity knob (e.g. 0.5x-2x sweep).
    combiner : "geometric_mean" (default, standard pedigree-matrix practice)
        or "arithmetic_mean" or "max" (most-conservative-dimension-dominates)
        or "weighted_mean" (requires `weights`).
    direction : "higher_is_better" (default, matching the workbook: scale_max is
        the highest confidence) or "lower_is_better" (flips the mapping).
        The default follows the convention actually used in the source data, so
        that a caller omitting it does not silently invert every sigma.
    weights : optional (w_coverage, w_frequency, w_spatial) for
        combiner="weighted_mean"; ignored otherwise.
    """
    dims = [coverage, frequency, spatial_boundary]
    valid_idx = [i for i, d in enumerate(dims) if d is not None and not (isinstance(d, float) and np.isnan(d))]
    if not valid_idx:
        return sigma_rel_worst * global_multiplier

    def _dim_penalty(d: float) -> float:
        d_clamped = max(1.0, min(scale_max, d))
        frac = (d_clamped - 1.0) / (scale_max - 1.0)  # 0=best-scored value, 1=worst-scored value
        if direction == "higher_is_better":
            frac = 1.0 - frac
        return sigma_rel_best + frac * (sigma_rel_worst - sigma_rel_best)

    penalties = [_dim_penalty(dims[i]) for i in valid_idx]

    if combiner == "arithmetic_mean":
        rel = float(np.mean(penalties))
    elif combiner == "max":
        rel = float(np.max(penalties))
    elif combiner == "weighted_mean":
        w = weights or (1.0, 1.0, 1.0)
        w_valid = [w[i] for i in valid_idx]
        rel = float(np.average(penalties, weights=w_valid))
    else:  # "geometric_mean" default
        rel = float(np.exp(np.mean(np.log(penalties))))

    return rel * global_multiplier
