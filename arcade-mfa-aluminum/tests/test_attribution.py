"""Tests for constraint attribution.

These answer R1-5 / R2-9: which reconciled values are driven by observations and
which are carried by the prior or by structure. A wrong attribution would be
worse than none, since it would misrepresent the evidential basis of a result.
"""

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.attribution import (
    classify_flows,
    observation_inventory,
    precision_shares,
    prior_posterior_comparison,
)


def test_shares_sum_to_one_and_split_correctly():
    q_obs = np.array([4.0, 0.0, 1.0])
    Q_prior = np.diag([1.0, 2.0, 1.0])
    soft = {"mass_balance": np.diag([0.0, 2.0, 2.0])}

    s = precision_shares(q_obs, Q_prior, soft)
    cols = [c for c in s.columns if c.startswith("share_")]
    assert np.allclose(s[cols].sum(axis=1), 1.0)

    # flow 0: 4 of 5 precision from observations
    assert s.loc[0, "share_observations"] == np.isclose(4 / 5, 0.8) or np.isclose(s.loc[0, "share_observations"], 0.8)
    # flow 1: unobserved -> zero observation share
    assert s.loc[1, "share_observations"] == 0.0
    # flow 2: evenly split prior / mass balance, no majority to observations
    assert np.isclose(s.loc[2, "share_observations"], 0.25)


def test_unobserved_flow_has_zero_observation_share():
    s = precision_shares(np.array([0.0, 5.0]), None, {"mass_balance": np.diag([1.0, 1.0])})
    assert s.loc[0, "share_observations"] == 0.0
    assert s.loc[1, "share_observations"] > 0.5


def test_inventory_counts_sources_and_spread():
    obs = pd.DataFrame({
        "flow_idx": [0, 0, 2],
        "source": ["USGS value 1", "Other literature", "USGS value 1"],
        "value": [100.0, 120.0, 50.0],
    })
    inv = observation_inventory(obs, n_flows=3)
    assert inv.loc[0, "n_observations"] == 2
    assert "Other literature" in inv.loc[0, "sources"]
    assert inv.loc[0, "obs_spread_pct"] > 0
    assert inv.loc[1, "n_observations"] == 0      # flow 1 unobserved
    assert inv.loc[2, "obs_spread_pct"] == 0.0    # single source, no spread


def test_contraction_detects_a_prior_dominated_flow():
    rng = np.random.default_rng(0)
    prior_mean = np.array([100.0, 100.0])
    prior_cov = np.diag([100.0, 100.0])           # prior sd 10 for both
    # flow 0 stays at prior width; flow 1 is sharply narrowed by data
    S = np.column_stack([
        rng.normal(100.0, 10.0, 20000),
        rng.normal(130.0, 1.0, 20000),
    ])
    cmp = prior_posterior_comparison(S, prior_mean, prior_cov)
    assert abs(cmp.loc[0, "contraction"]) < 0.1        # barely contracted
    assert cmp.loc[1, "contraction"] > 0.95            # strongly contracted
    assert abs(cmp.loc[1, "shift_in_prior_sd"] - 3.0) < 0.2


def test_classification_labels_each_regime():
    shares = precision_shares(
        np.array([10.0, 0.0, 0.0, 10.0]),
        np.diag([1.0, 10.0, 1.0, 1.0]),
        {"mass_balance": np.diag([1.0, 1.0, 10.0, 1.0])},
    )
    inv = observation_inventory(
        pd.DataFrame({"flow_idx": [0, 3, 3], "source": ["a", "a", "b"],
                      "value": [1.0, 2.0, 3.0]}),
        n_flows=4,
    )
    out = classify_flows(shares, inv)
    got = dict(zip(out.flow_idx, out.determined_by))
    assert got[0] == "observation-driven"
    assert got[1] == "prior-informed"
    assert got[2] == "structure-determined"
    assert got[3] == "multi-source"
