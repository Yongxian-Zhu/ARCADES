"""Smoke tests for arcade_mfa_aluminum.graph -- run with: pytest tests/test_graph.py"""

import numpy as np
import pandas as pd

from arcade_mfa_aluminum.graph import build_node_balance, nullspace, project_to_mass_balance


def _toy_network():
    # 3 flows: 0: source->A, 1: A->B, 2: B->sink. Node A and B are internal.
    df = pd.DataFrame({
        "flow_idx": [0, 1, 2],
        "from_node": [100, 1, 2],
        "to_node": [1, 2, 200],
    })
    return df


def test_build_node_balance_shapes():
    df = _toy_network()
    mb = build_node_balance(df, n_flows=3)
    assert mb.A.shape == (2, 3)   # 2 internal nodes (1 and 2), 3 flows
    assert set(mb.nodes) == {1, 2}


def test_nullspace_dimension():
    df = _toy_network()
    mb = build_node_balance(df, n_flows=3)
    Z = nullspace(mb.A)
    # a simple chain has rank(A) = 2 -> nullspace dimension = 3 - 2 = 1
    assert Z.shape == (3, 1)


def test_project_to_mass_balance_satisfies_constraints():
    df = _toy_network()
    mb = build_node_balance(df, n_flows=3)
    x = np.array([10.0, 4.0, 20.0])   # violates balance (should be 10,10,10)
    x_proj = project_to_mass_balance(x, mb.A)
    residual = mb.A @ x_proj
    assert np.allclose(residual, 0.0, atol=1e-6)


def test_empty_constraints_is_noop():
    A = np.zeros((0, 5))
    x = np.arange(5.0)
    assert np.allclose(project_to_mass_balance(x, A), x)
    assert nullspace(A).shape == (5, 5)


def test_build_bounds_never_allows_negative_flows():
    """A zero-valued observation must not be given a negative lower bound.

    The old degeneracy guard fixed lb == ub by lowering lb to -1e-6, which let
    the sampler return negative mass flows (22 of them in the 2017 run).
    """
    import numpy as np
    from arcade_mfa_aluminum.pipeline import _build_bounds

    y = np.array([0.0, 100.0, 0.0, 5.0, np.nan])
    lb, ub = _build_bounds(y, {"bounds": {"observed_lower_frac": 0.2,
                                          "observed_upper_frac": 1.8}})
    assert (lb >= 0).all(), f"negative lower bound: {lb}"
    assert (ub > lb).all(), "degenerate interval survived the guard"
    assert (ub - lb >= 1e-6 - 1e-15).all()
    # Ordinary observed flows keep the configured fractional bounds.
    assert np.isclose(lb[1], 20.0) and np.isclose(ub[1], 180.0)
