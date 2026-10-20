#!/usr/bin/env python3
"""
indicators.py
Compute circularity and supply-chain indicators from posterior samples.

Indicators (per posterior draw):
  • Collection rate          = EOL_collected / EOL_generated
  • EOL recycling rate       = EOL_recycled_domestic / EOL_generated
  • Domestic retention       = EOL_recycled_domestic / EOL_collected
  • Recycled content rate    = secondary_input / total_fabrication_input
  • Net import reliance      = (imports − exports) / apparent_consumption

All are reported as posterior median ± 95 % credible interval.
"""

import os
import numpy as np
import pandas as pd
import arviz as az

from io_utils import save_csv, ensure_dir

# ── flow-index mapping (must match system definition) ───────────────
# These are the default Flow index values from the SI flow catalog.
# Adjust if your flow_data.csv uses different numbering.

# EOL scrap generated (total) — sum of sectoral EOL → EOL pool
EOL_SECTOR_FLOWS = list(range(234, 242))   # Flows 234–241 (0-based: 233–240)

# EOL scrap → export (Flow 35)
EOL_EXPORT_FLOW = 35

# EOL scrap → loss / disposal (Flow 36)
EOL_LOSS_FLOW = 36

# EOL scrap → collected for recycling (Flow 37)
EOL_COLLECTED_FLOW = 37

# Collected EOL → remelting (Flow 38)
EOL_TO_REMELTING = 38

# Collected EOL → refining (Flow 39)
EOL_TO_REFINING = 39

# Forming scrap recycled → remelting (42) + refining (43)
FORMING_TO_REMELT = 42
FORMING_TO_REFINE = 43

# Fabrication scrap recycled → remelting (46) + refining (47)
FAB_TO_REMELT = 46
FAB_TO_REFINE = 47

# Primary smelting → remelting (12) + refining (13)
PRIMARY_TO_REMELT = 12
PRIMARY_TO_REFINE = 13

# Import unwrought → wrought ingot (14) + refining (15)
IMPORT_UNWROUGHT_WROUGHT = 14
IMPORT_UNWROUGHT_REFINE = 15


def _idx(flow_id: int, offset: int = 1) -> int:
    """Convert 1-based Flow index to 0-based array position."""
    return flow_id - offset


def compute_indicators(idata, flow_offset: int = 1):
    """Compute indicators for every posterior draw.

    Parameters
    ----------
    idata : arviz.InferenceData with posterior["x"]
    flow_offset : 1 if Flow index is 1-based, 0 if 0-based

    Returns
    -------
    pd.DataFrame with one row per draw and columns for each indicator.
    """
    post = idata.posterior["x"]
    # shape: (chain, draw, flow)
    x = post.stack(sample=("chain", "draw")).transpose(
        "flow", "sample").to_numpy()  # (n_flows, n_samples)

    n_samples = x.shape[1]
    idx = lambda fid: _idx(fid, flow_offset)

    # EOL generated = sum of sectoral flows
    eol_gen = np.sum([x[idx(f)] for f in EOL_SECTOR_FLOWS], axis=0)

    # EOL collected for recycling
    eol_coll = x[idx(EOL_COLLECTED_FLOW)]

    # EOL exported
    eol_exp = x[idx(EOL_EXPORT_FLOW)]

    # EOL recycled domestically (to remelting + refining)
    eol_rec_dom = x[idx(EOL_TO_REMELTING)] + x[idx(EOL_TO_REFINING)]

    # Total secondary input to remelting/refining
    sec_input = (x[idx(EOL_TO_REMELTING)] + x[idx(EOL_TO_REFINING)]
                 + x[idx(FORMING_TO_REMELT)] + x[idx(FORMING_TO_REFINE)]
                 + x[idx(FAB_TO_REMELT)] + x[idx(FAB_TO_REFINE)])

    # Total primary input
    prim_input = (x[idx(PRIMARY_TO_REMELT)] + x[idx(PRIMARY_TO_REFINE)]
                  + x[idx(IMPORT_UNWROUGHT_WROUGHT)]
                  + x[idx(IMPORT_UNWROUGHT_REFINE)])

    total_input = sec_input + prim_input

    eps = 1e-12
    collection_rate = eol_coll / np.maximum(eol_gen, eps)
    eol_rr = eol_rec_dom / np.maximum(eol_gen, eps)
    dom_retention = eol_rec_dom / np.maximum(eol_coll, eps)
    recycled_content = sec_input / np.maximum(total_input, eps)

    df = pd.DataFrame({
        "collection_rate": collection_rate,
        "eol_recycling_rate": eol_rr,
        "domestic_retention": dom_retention,
        "recycled_content": recycled_content,
        "eol_generated": eol_gen,
        "eol_collected": eol_coll,
        "eol_exported": eol_exp,
        "eol_recycled_domestic": eol_rec_dom,
    })
    return df


def summarise_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Posterior median and 95 % CI for each indicator."""
    metrics = ["collection_rate", "eol_recycling_rate",
               "domestic_retention", "recycled_content",
               "eol_generated", "eol_collected", "eol_exported",
               "eol_recycled_domestic"]
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        vals = df[m].to_numpy()
        rows.append(dict(
            indicator=m,
            median=float(np.median(vals)),
            mean=float(np.mean(vals)),
            ci_2_5=float(np.percentile(vals, 2.5)),
            ci_97_5=float(np.percentile(vals, 97.5)),
            std=float(np.std(vals)),
        ))
    return pd.DataFrame(rows)


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace",
                    default="pymc_full_space_res_2017/"
                            "pymc_full_space_res_2017.nc")
    ap.add_argument("--output_dir", default="pymc_full_space_res_2017")
    ap.add_argument("--flow_offset", type=int, default=1,
                    help="1 if Flow index is 1-based, 0 if 0-based")
    args = ap.parse_args()

    idata = az.from_netcdf(args.trace)
    draws = compute_indicators(idata, flow_offset=args.flow_offset)
    summ = summarise_indicators(draws)

    ensure_dir(args.output_dir)
    save_csv(draws, os.path.join(args.output_dir, "indicator_draws.csv"))
    save_csv(summ, os.path.join(args.output_dir, "indicator_summary.csv"))
    print(summ.to_string(index=False))