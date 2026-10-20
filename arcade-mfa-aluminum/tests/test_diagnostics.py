"""Tests for the convergence diagnostics.

These have to be right in BOTH directions: a diagnostic that never fires is as
useless as one that always does, so each estimator is checked on chains that
should pass and on chains that should fail.
"""

import numpy as np
import pytest

from arcade_mfa_aluminum.diagnostics import (
    convergence_summary,
    effective_sample_size,
    split_rhat,
)


def test_rhat_near_one_for_independent_chains():
    rng = np.random.default_rng(1)
    chains = rng.standard_normal((4, 4000, 3))
    r = split_rhat(chains)
    assert (r < 1.01).all(), f"iid chains flagged as unconverged: {r}"


def test_rhat_detects_chains_with_different_means():
    """The failure mode that matters: chains exploring different regions."""
    rng = np.random.default_rng(2)
    chains = rng.standard_normal((4, 2000, 1))
    chains[0] += 5.0                      # one chain stuck elsewhere
    r = split_rhat(chains)
    assert r[0] > 1.5, f"Rhat failed to flag a badly offset chain: {r[0]}"


def test_rhat_detects_within_chain_drift():
    """Split-Rhat exists to catch drift a whole-chain comparison would miss."""
    rng = np.random.default_rng(3)
    n = 4000
    trend = np.linspace(0.0, 6.0, n)
    chains = rng.standard_normal((4, n, 1)) + trend[None, :, None]
    assert split_rhat(chains)[0] > 1.2


def test_rhat_handles_constant_parameter():
    chains = np.full((4, 500, 2), 3.0)
    r = split_rhat(chains)
    assert np.allclose(r, 1.0) and np.isfinite(r).all()


def test_ess_near_n_for_independent_draws():
    rng = np.random.default_rng(4)
    n_chains, n_draws = 4, 2000
    chains = rng.standard_normal((n_chains, n_draws, 2))
    es = effective_sample_size(chains)
    total = n_chains * n_draws
    assert (es > 0.7 * total).all(), f"iid ESS far below n: {es}"
    assert (es < 1.5 * total).all()


def test_ess_collapses_for_autocorrelated_chains():
    """AR(1) with rho=0.95 has ESS roughly (1-rho)/(1+rho) ~ 2.6% of n."""
    rng = np.random.default_rng(5)
    n_chains, n_draws, rho = 4, 4000, 0.95
    x = np.zeros((n_chains, n_draws, 1))
    for c in range(n_chains):
        for t in range(1, n_draws):
            x[c, t, 0] = rho * x[c, t-1, 0] + rng.standard_normal()
    es = effective_sample_size(x)[0]
    total = n_chains * n_draws
    assert es < 0.15 * total, f"ESS did not collapse for rho=0.95: {es}/{total}"
    assert es > 0.005 * total


def test_convergence_summary_shape_and_flagging():
    rng = np.random.default_rng(6)
    chains = rng.standard_normal((4, 1000, 5))
    chains[:, :, 4] += np.array([0.0, 8.0, 0.0, 0.0])[:, None]   # param 4 broken

    df = convergence_summary(chains, labels=np.arange(100, 105))
    assert list(df.columns) == [
        "flow_idx", "rhat", "ess", "mean", "sd", "q2.5", "q97.5", "converged"
    ]
    assert len(df) == 5
    assert (df["flow_idx"].to_numpy() == np.arange(100, 105)).all()
    assert df.loc[df["flow_idx"] == 104, "converged"].item() is np.False_ or \
           not bool(df.loc[df["flow_idx"] == 104, "converged"].item())
    assert bool(df.loc[df["flow_idx"] == 100, "converged"].item())
    assert (df["q2.5"] < df["q97.5"]).all()


# ---------------------------------------------------------------------------
# Config keys that must not be silently ignored
# ---------------------------------------------------------------------------

def test_log_space_raises_instead_of_silently_running_linear(tmp_path):
    """formulation.space='log' must raise rather than fall through to the
    LINEAR formulation. For a paper, silently reporting a different
    formulation than the one configured is worse than failing outright.
    """
    import yaml
    from arcade_mfa_aluminum.pipeline import run

    cfg = {
        "run": {"name": "t", "year": 2017},
        "data": {"source_name": "2017 data"},
        "formulation": {"type": "flow_based", "space": "log"},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(NotImplementedError, match="log"):
        run(str(p))


def test_unknown_space_raises(tmp_path):
    import yaml
    from arcade_mfa_aluminum.pipeline import run

    cfg = {
        "run": {"name": "t", "year": 2017},
        "data": {"source_name": "2017 data"},
        "formulation": {"type": "flow_based", "space": "polar"},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="formulation.space"):
        run(str(p))
