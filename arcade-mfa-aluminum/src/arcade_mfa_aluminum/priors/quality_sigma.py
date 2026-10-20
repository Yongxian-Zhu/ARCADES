"""
arcade_mfa_aluminum.priors.quality_sigma
---------------------------------
Convert the workbook's per-observation pedigree scores into observation sigma.

The "2017 data" sheet carries three confidence dimensions per value (coverage,
frequency, spatial boundary). `adapters.aluminum.adapter.pedigree_to_rel_sigma`
holds the combiner; this module applies it across all flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def sigma_obs_from_pedigree(
    values: np.ndarray,
    coverage: np.ndarray,
    frequency: np.ndarray,
    spatial_boundary: np.ndarray,
    *,
    mapping: dict | None = None,
) -> np.ndarray:
    """Vectorized sigma from the authoritative workbook's 3-dimension
    pedigree columns. Thin wrapper around
    `adapters.aluminum.adapter.pedigree_to_rel_sigma` to keep one canonical
    combiner implementation.

    `mapping` is an optional dict of kwargs forwarded to
    `pedigree_to_rel_sigma` (scale_max, sigma_rel_best, sigma_rel_worst,
    global_multiplier, combiner, direction, weights) -- this is the hook the
    sensitivity sweep script uses to test alternative mapping configurations
    without touching code. See `configs/pedigree_mapping.yaml` and
    `docs/pedigree_sensitivity.md`.
    """
    from arcade_mfa_aluminum.adapters.aluminum.adapter import pedigree_to_rel_sigma

    kwargs = mapping or {}
    rel = np.array([
        pedigree_to_rel_sigma(c, f, s, **kwargs)
        for c, f, s in zip(coverage, frequency, spatial_boundary)
    ])
    sigma = rel * np.maximum(np.abs(values), 1e-6)
    return np.maximum(sigma, 1e-3)


def sigma_for_observations(
    obs,
    *,
    mapping: dict | None = None,
    fallback_rel_sigma: float = 0.10,
    source_multipliers: dict | None = None,
    zero_value_scale: float = 1.0,
    floor: float = 1e-3,
):
    """Absolute sigma for every row of a long-form observation table.

    `obs` needs columns: source, value, coverage, frequency, spatial.

    Rows carrying a pedigree triple are mapped through
    `pedigree_to_rel_sigma`; rows without one fall back to
    `fallback_rel_sigma`. A per-source multiplier then scales the result, which
    is how relative trust between sources is expressed: a multiplier below 1
    tightens a source, above 1 loosens it. Weighting sources this way keeps the
    decision in configuration and visible, rather than implicit in which column
    happens to be read first.
    """
    from arcade_mfa_aluminum.adapters.aluminum.adapter import pedigree_to_rel_sigma

    kwargs = mapping or {}
    mult = source_multipliers or {}
    rel = np.empty(len(obs), dtype=float)
    for i, (_, r) in enumerate(obs.iterrows()):
        c, f, sb = r["coverage"], r["frequency"], r["spatial"]
        has_pedigree = not (pd.isna(c) and pd.isna(f) and pd.isna(sb))
        base = pedigree_to_rel_sigma(c, f, sb, **kwargs) if has_pedigree else fallback_rel_sigma
        rel[i] = base * float(mult.get(r["source"], 1.0))
    # A purely relative sigma collapses at zero, which would assert near-perfect
    # certainty on a reported zero regardless of its pedigree. Fall back to an
    # absolute scale there, so a poorly-evidenced zero stays looser than a
    # well-evidenced one.
    v = np.abs(obs["value"].to_numpy(dtype=float))
    sigma = np.where(v > 1e-9, rel * v, rel * zero_value_scale)
    return np.maximum(sigma, floor)
