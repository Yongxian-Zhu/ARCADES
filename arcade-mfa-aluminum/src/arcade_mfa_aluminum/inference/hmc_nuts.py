"""
arcade_mfa_aluminum.inference.hmc_nuts
--------------------------------------

HMC/NUTS posterior sampling for the bounded, linear-scale flow formulation.

The target posterior is

    p(x | y) proportional to

        product_i Normal(y_i | C_i x, sigma_obs_i)
        product_k Normal(b_k | A_k x, sigma_constraint_k)
        p_transfer(x)
        1{x_lb <= x <= x_ub},

where:

* x is the physical flow vector;
* C maps flows to direct or aggregate observations;
* A contains soft mass-balance or other linear-Gaussian constraints;
* p_transfer is an optional multivariate Gaussian prior; and
* the indicator defines exact lower and upper support bounds.

Bounds are represented using PyMC Uniform variables. PyMC transforms these
bounded variables to unconstrained coordinates for HMC/NUTS and automatically
includes the transformation Jacobian.

This implementation is intended for soft mass balance. Do not approximate exact
mass balance using an extremely small Gaussian standard deviation. Exact mass
balance requires a reduced-coordinate or null-space implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from arcade_mfa_aluminum.graph import MassBalanceSystem


@dataclass(frozen=True)
class GaussianLinearBlock:
    """Linear-Gaussian posterior block.

    The block defines

        target ~ Normal(matrix @ flows, sigma).

    Examples
    --------
    Direct flow observations:
        Each matrix row contains one coefficient equal to one.

    Aggregate observations:
        A row may contain several coefficients.

    Soft mass balance:
        matrix is the node-balance matrix and target is normally zero.

    Linearized allocation constraints:
        matrix contains the linearized allocation relationships.
    """

    name: str
    matrix: np.ndarray
    target: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class TransferredGaussianPrior:
    """Optional multivariate Gaussian prior on the flow vector.

    Parameters
    ----------
    mean
        Prior mean vector.

    covariance
        Prior covariance matrix.

    covariance_inflation
        Multiplier applied to the covariance matrix. For example, 1.5
        increases variances by 50% and standard deviations by sqrt(1.5).

    jitter
        Relative diagonal regularization used to make a numerically singular
        sample covariance positive definite.
    """

    mean: np.ndarray
    covariance: np.ndarray
    covariance_inflation: float = 1.0
    jitter: float = 1e-8


@dataclass(frozen=True)
class HMCSamplerSettings:
    """Configuration for NUTS or fixed-trajectory HMC."""

    method: Literal["nuts", "hmc"] = "nuts"

    n_chains: int = 4
    n_draws: int = 2000
    n_tune: int = 2000
    seed: int = 42
    cores: int | None = None

    target_accept: float = 0.95

    # NUTS-specific setting.
    max_treedepth: int = 12

    # Fixed-trajectory HMC settings.
    path_length: float = 2.0
    step_scale: float = 0.25

    progressbar: bool = True
    compute_convergence_checks: bool = True


@dataclass
class HMCSamplerResult:
    """Results returned by :func:`sample_hmc_nuts`."""

    samples: np.ndarray
    # Shape: (n_chains, n_draws, n_flows)

    x_map: np.ndarray
    method: str
    inference_data: object
    summary: object
    diagnostics: dict = field(default_factory=dict)


def _as_finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    """Convert an input to a finite one-dimensional float array."""

    array = np.asarray(value, dtype=float).reshape(-1)

    if not np.all(np.isfinite(array)):
        bad = np.flatnonzero(~np.isfinite(array))
        raise ValueError(
            f"{name} contains non-finite values at positions "
            f"{bad[:20].tolist()}"
        )

    return array


def _validate_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite lower and upper flow bounds."""

    lower = _as_finite_vector(lower, "lower")
    upper = _as_finite_vector(upper, "upper")

    if lower.size != upper.size:
        raise ValueError(
            f"lower has length {lower.size}, but upper has length "
            f"{upper.size}"
        )

    if lower.size == 0:
        raise ValueError("At least one flow is required")

    invalid = np.flatnonzero(lower >= upper)
    if invalid.size:
        raise ValueError(
            "Every sampled flow must satisfy lower < upper. Invalid flow "
            f"indices: {invalid[:20].tolist()}"
        )

    return lower, upper


def _validate_block(
    block: GaussianLinearBlock,
    n_flows: int,
) -> GaussianLinearBlock:
    """Validate and normalize a linear-Gaussian block."""

    matrix = np.asarray(block.matrix, dtype=float)
    target = _as_finite_vector(block.target, f"{block.name}.target")
    sigma = _as_finite_vector(block.sigma, f"{block.name}.sigma")

    if matrix.ndim != 2:
        raise ValueError(
            f"{block.name}.matrix must be two-dimensional; "
            f"got shape {matrix.shape}"
        )

    if matrix.shape[1] != n_flows:
        raise ValueError(
            f"{block.name}.matrix has {matrix.shape[1]} columns, but "
            f"n_flows={n_flows}"
        )

    if matrix.shape[0] != target.size:
        raise ValueError(
            f"{block.name}.matrix has {matrix.shape[0]} rows, but "
            f"{block.name}.target has length {target.size}"
        )

    if sigma.size == 1 and target.size != 1:
        sigma = np.full(target.size, sigma.item(), dtype=float)

    if sigma.size != target.size:
        raise ValueError(
            f"{block.name}.sigma has length {sigma.size}, but target has "
            f"length {target.size}"
        )

    if np.any(sigma <= 0.0):
        bad = np.flatnonzero(sigma <= 0.0)
        raise ValueError(
            f"{block.name}.sigma must be strictly positive. Invalid rows: "
            f"{bad[:20].tolist()}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{block.name}.matrix contains non-finite values")

    return GaussianLinearBlock(
        name=block.name,
        matrix=matrix,
        target=target,
        sigma=sigma,
    )


def _validate_settings(settings: HMCSamplerSettings) -> None:
    """Validate sampler settings before constructing the PyMC model."""

    if settings.method not in {"nuts", "hmc"}:
        raise ValueError(
            f"Unknown sampling method {settings.method!r}; expected "
            "'nuts' or 'hmc'"
        )

    if settings.n_chains <= 0:
        raise ValueError("n_chains must be positive")

    if settings.n_draws <= 0:
        raise ValueError("n_draws must be positive")

    if settings.n_tune < 0:
        raise ValueError("n_tune cannot be negative")

    if not 0.0 < settings.target_accept < 1.0:
        raise ValueError("target_accept must lie strictly between zero and one")

    if settings.max_treedepth <= 0:
        raise ValueError("max_treedepth must be positive")

    if settings.path_length <= 0.0:
        raise ValueError("path_length must be positive")

    if settings.step_scale <= 0.0:
        raise ValueError("step_scale must be positive")


def _prepare_prior(
    prior: TransferredGaussianPrior,
    n_flows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and regularize a transferred Gaussian prior."""

    mean = _as_finite_vector(prior.mean, "prior.mean")
    covariance = np.asarray(prior.covariance, dtype=float)

    if mean.size != n_flows:
        raise ValueError(
            f"prior.mean has length {mean.size}, but n_flows={n_flows}"
        )

    if covariance.shape != (n_flows, n_flows):
        raise ValueError(
            f"prior.covariance has shape {covariance.shape}; expected "
            f"({n_flows}, {n_flows})"
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError("prior.covariance contains non-finite values")

    if prior.covariance_inflation <= 0.0:
        raise ValueError("prior.covariance_inflation must be positive")

    if prior.jitter < 0.0:
        raise ValueError("prior.jitter cannot be negative")

    covariance = 0.5 * (covariance + covariance.T)
    covariance = prior.covariance_inflation * covariance

    diagonal_scale = max(
        float(np.mean(np.maximum(np.diag(covariance), 0.0))),
        1.0,
    )

    covariance = (
        covariance
        + prior.jitter * diagonal_scale * np.eye(n_flows)
    )

    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        eigenvalues = np.linalg.eigvalsh(covariance)
        raise ValueError(
            "Transferred prior covariance is not positive definite after "
            "regularization. Minimum eigenvalue is "
            f"{float(eigenvalues.min()):.6g}."
        ) from exc

    return mean, chol


def _move_strictly_inside_bounds(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_margin: float = 1e-6,
) -> np.ndarray:
    """Move an initialization vector strictly inside its support."""

    value = _as_finite_vector(value, "x_map")

    if value.size != lower.size:
        raise ValueError(
            f"x_map has length {value.size}, expected {lower.size}"
        )

    width = upper - lower
    margin = np.maximum(relative_margin * width, 1e-10)

    return np.clip(
        value,
        lower + margin,
        upper - margin,
    )


def _require_unique_block_names(
    blocks: Sequence[GaussianLinearBlock],
) -> None:
    """Require unique block names because they become PyMC variable names."""

    names = [block.name for block in blocks]
    duplicates = sorted(
        name for name in set(names) if names.count(name) > 1
    )

    if duplicates:
        raise ValueError(
            "Gaussian block names must be unique. Duplicate names: "
            f"{duplicates}"
        )


def direct_observation_matrix(
    n_flows: int,
    observed_flow_idx: np.ndarray,
) -> np.ndarray:
    """Construct a one-hot matrix for direct flow observations.

    Repeated flow indices are permitted. This supports multiple independent
    observations of the same latent flow.
    """

    if n_flows <= 0:
        raise ValueError("n_flows must be positive")

    raw_idx = np.asarray(observed_flow_idx)

    if raw_idx.ndim == 0:
        raw_idx = raw_idx.reshape(1)

    raw_idx = raw_idx.reshape(-1)

    try:
        numeric_idx = raw_idx.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "observed_flow_idx contains nonnumeric values"
        ) from exc

    if not np.all(np.isfinite(numeric_idx)):
        raise ValueError(
            "observed_flow_idx contains non-finite values"
        )

    if not np.all(numeric_idx == np.floor(numeric_idx)):
        raise ValueError(
            "observed_flow_idx must contain integer-valued indices"
        )

    observed_flow_idx = numeric_idx.astype(int)

    if observed_flow_idx.size == 0:
        return np.zeros((0, n_flows), dtype=float)

    invalid = np.flatnonzero(
        (observed_flow_idx < 0)
        | (observed_flow_idx >= n_flows)
    )

    if invalid.size:
        raise ValueError(
            "Observed flow indices must be in "
            f"[0, {n_flows - 1}]. Invalid entries occur at positions "
            f"{invalid[:20].tolist()}."
        )

    matrix = np.zeros(
        (observed_flow_idx.size, n_flows),
        dtype=float,
    )

    matrix[
        np.arange(observed_flow_idx.size),
        observed_flow_idx,
    ] = 1.0

    return matrix


def make_direct_observation_block(
    *,
    n_flows: int,
    observed_flow_idx: np.ndarray,
    observed_values: np.ndarray,
    observed_sigmas: np.ndarray,
    name: str = "flow_observations",
) -> GaussianLinearBlock:
    """Create a Gaussian block for direct flow observations."""

    observed_values = _as_finite_vector(
        observed_values,
        "observed_values",
    )
    observed_sigmas = _as_finite_vector(
        observed_sigmas,
        "observed_sigmas",
    )

    matrix = direct_observation_matrix(
        n_flows,
        observed_flow_idx,
    )

    if matrix.shape[0] != observed_values.size:
        raise ValueError(
            f"observed_flow_idx contains {matrix.shape[0]} entries, but "
            f"observed_values contains {observed_values.size}"
        )

    if observed_sigmas.size == 1 and observed_values.size != 1:
        observed_sigmas = np.full(
            observed_values.size,
            observed_sigmas.item(),
            dtype=float,
        )

    if observed_sigmas.size != observed_values.size:
        raise ValueError(
            f"observed_sigmas has length {observed_sigmas.size}, but "
            f"observed_values has length {observed_values.size}"
        )

    return GaussianLinearBlock(
        name=name,
        matrix=matrix,
        target=observed_values,
        sigma=observed_sigmas,
    )


def make_soft_mass_balance_block(
    mass_balance: MassBalanceSystem,
    node_totals: Mapping[int, float],
    *,
    rel_sigma: float,
    min_total: float = 1e-6,
    name: str = "soft_mass_balance",
) -> GaussianLinearBlock | None:
    """Create a Gaussian soft mass-balance block.

    For retained node ``p``,

        (A x)_p ~ Normal(b_p, sigma_p^2)

    with

        sigma_p = rel_sigma * T_p,

    where ``T_p`` is the reference node throughput.

    Nodes with negligible throughput are omitted because a zero or nearly zero
    standard deviation would unintentionally approximate a hard equality.
    """

    if rel_sigma <= 0.0:
        raise ValueError(
            f"mass_balance.rel_sigma must be positive; got {rel_sigma}"
        )

    if min_total < 0.0:
        raise ValueError(
            f"min_total cannot be negative; got {min_total}"
        )

    A = np.asarray(mass_balance.A, dtype=float)
    b = np.asarray(mass_balance.b, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError(
            f"mass_balance.A must be two-dimensional; got {A.shape}"
        )

    if A.shape[0] != b.size:
        raise ValueError(
            f"mass_balance.A has {A.shape[0]} rows, but "
            f"mass_balance.b has length {b.size}"
        )

    if A.shape[0] != len(mass_balance.nodes):
        raise ValueError(
            f"mass_balance.A has {A.shape[0]} rows, but nodes contains "
            f"{len(mass_balance.nodes)} entries"
        )

    keep: list[int] = []
    sigma: list[float] = []

    for row_index, node in enumerate(mass_balance.nodes):
        total = float(node_totals.get(int(node), 0.0))

        if not np.isfinite(total):
            raise ValueError(
                f"Reference throughput for node {node} is not finite"
            )

        if total > min_total:
            keep.append(row_index)
            sigma.append(rel_sigma * total)

    if not keep:
        return None

    return GaussianLinearBlock(
        name=name,
        matrix=A[keep],
        target=b[keep],
        sigma=np.asarray(sigma, dtype=float),
    )


def build_flow_blocks(
    *,
    n_flows: int,
    observed_flow_idx: np.ndarray,
    observed_values: np.ndarray,
    observed_sigmas: np.ndarray,
    mass_balance: MassBalanceSystem,
    node_totals: Mapping[int, float],
    mass_balance_rel_sigma: float,
    aggregate_blocks: Sequence[GaussianLinearBlock] = (),
    allocation_blocks: Sequence[GaussianLinearBlock] = (),
) -> list[GaussianLinearBlock]:
    """Build the standard flow-model likelihood and constraint blocks."""

    blocks: list[GaussianLinearBlock] = []

    observed_values_array = np.asarray(observed_values).reshape(-1)

    if observed_values_array.size:
        blocks.append(
            make_direct_observation_block(
                n_flows=n_flows,
                observed_flow_idx=observed_flow_idx,
                observed_values=observed_values,
                observed_sigmas=observed_sigmas,
            )
        )

    mass_balance_block = make_soft_mass_balance_block(
        mass_balance,
        node_totals,
        rel_sigma=mass_balance_rel_sigma,
    )

    if mass_balance_block is not None:
        blocks.append(mass_balance_block)

    blocks.extend(aggregate_blocks)
    blocks.extend(allocation_blocks)

    return blocks


def sample_hmc_nuts(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    blocks: Sequence[GaussianLinearBlock],
    x_map: np.ndarray,
    prior: TransferredGaussianPrior | None = None,
    settings: HMCSamplerSettings | None = None,
) -> HMCSamplerResult:
    """Sample the bounded flow posterior using HMC or NUTS.

    Parameters
    ----------
    lower, upper
        Finite lower and upper bounds for every flow.

    blocks
        Observation, soft mass-balance, and other linear-Gaussian blocks.

    x_map
        MAP flow estimate used to initialize sampling. It does not define the
        center of the posterior unless separately included as a prior.

    prior
        Optional transferred multivariate Gaussian prior.

    settings
        HMC or NUTS settings.

    Returns
    -------
    HMCSamplerResult
        Samples, ArviZ output, summary table, and diagnostics.
    """

    # Imports are deferred so the package can still use non-PyMC backends when
    # PyMC is not installed.
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:
        raise ImportError(
            "HMC/NUTS sampling requires pymc, pytensor, and arviz. "
            "Install the optional dependencies before selecting "
            "inference.method: nuts or inference.method: hmc."
        ) from exc

    settings = settings or HMCSamplerSettings()
    _validate_settings(settings)

    lower, upper = _validate_bounds(lower, upper)
    n_flows = lower.size

    validated_blocks = [
        _validate_block(block, n_flows)
        for block in blocks
        if np.asarray(block.target).size > 0
    ]
    _require_unique_block_names(validated_blocks)

    initial_value = _move_strictly_inside_bounds(
        x_map,
        lower,
        upper,
    )

    prior_parameters = None
    if prior is not None:
        prior_parameters = _prepare_prior(prior, n_flows)

    coords = {
        "flow": np.arange(n_flows, dtype=int),
    }

    with pm.Model(coords=coords) as model:
        # This contributes constant density in physical flow space within the
        # specified support. PyMC applies an interval transform internally.
        flows = pm.Uniform(
            "flows",
            lower=lower,
            upper=upper,
            initval=initial_value,
            dims="flow",
        )

        for block_number, block in enumerate(validated_blocks):
            row_dimension = f"{block.name}_row_{block_number}"

            model.add_coord(
                row_dimension,
                np.arange(block.target.size, dtype=int),
            )

            matrix_tensor = pt.as_tensor_variable(block.matrix)

            pm.Normal(
                block.name,
                mu=pt.dot(matrix_tensor, flows),
                sigma=block.sigma,
                observed=block.target,
                dims=row_dimension,
            )

        if prior_parameters is not None:
            prior_mean, prior_chol = prior_parameters

            transferred_distribution = pm.MvNormal.dist(
                mu=prior_mean,
                chol=prior_chol,
            )

            # Uniform supplies bounded support. This Potential multiplies that
            # support by the transferred Gaussian density.
            pm.Potential(
                "transferred_flow_prior",
                pm.logp(transferred_distribution, flows),
            )

        if settings.method == "nuts":
            step = pm.NUTS(
                target_accept=settings.target_accept,
                max_treedepth=settings.max_treedepth,
            )
        else:
            step = pm.HamiltonianMC(
                target_accept=settings.target_accept,
                path_length=settings.path_length,
                step_scale=settings.step_scale,
            )

        inference_data = pm.sample(
            draws=settings.n_draws,
            tune=settings.n_tune,
            chains=settings.n_chains,
            cores=settings.cores or settings.n_chains,
            step=step,
            initvals={"flows": initial_value},
            random_seed=settings.seed,
            return_inferencedata=True,
            progressbar=settings.progressbar,
            compute_convergence_checks=settings.compute_convergence_checks,
        )

    samples = np.asarray(
        inference_data.posterior["flows"].values,
        dtype=float,
    )

    expected_shape = (
        settings.n_chains,
        settings.n_draws,
        n_flows,
    )

    if samples.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected posterior sample shape {samples.shape}; expected "
            f"{expected_shape}"
        )

    summary = az.summary(
        inference_data,
        var_names=["flows"],
        round_to=None,
    )

    sample_stats = inference_data.sample_stats

    n_divergences = 0
    if "diverging" in sample_stats:
        n_divergences = int(
            np.asarray(sample_stats["diverging"].values).sum()
        )

    diagnostics = {
        "method": settings.method,
        "n_chains": settings.n_chains,
        "n_draws_per_chain": settings.n_draws,
        "n_tune_per_chain": settings.n_tune,
        "n_samples_total": int(
            settings.n_chains * settings.n_draws
        ),
        "n_divergences": n_divergences,
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "max_lower_bound_violation": float(
            np.maximum(
                lower[None, None, :] - samples,
                0.0,
            ).max()
        ),
        "max_upper_bound_violation": float(
            np.maximum(
                samples - upper[None, None, :],
                0.0,
            ).max()
        ),
    }

    if "tree_depth" in sample_stats:
        tree_depth = np.asarray(
            sample_stats["tree_depth"].values
        )

        diagnostics["max_tree_depth_observed"] = int(
            tree_depth.max()
        )
        diagnostics["n_max_treedepth"] = int(
            (tree_depth >= settings.max_treedepth).sum()
        )

    if "acceptance_rate" in sample_stats:
        diagnostics["mean_acceptance_rate"] = float(
            np.asarray(
                sample_stats["acceptance_rate"].values
            ).mean()
        )

    if "step_size" in sample_stats:
        diagnostics["mean_step_size"] = float(
            np.asarray(sample_stats["step_size"].values).mean()
        )

    return HMCSamplerResult(
        samples=samples,
        x_map=np.asarray(x_map, dtype=float),
        method=settings.method,
        inference_data=inference_data,
        summary=summary,
        diagnostics=diagnostics,
    )


def run_flow_hmc_nuts(
    *,
    n_flows: int,
    observed_flow_idx: np.ndarray,
    observed_values: np.ndarray,
    observed_sigmas: np.ndarray,
    mass_balance: MassBalanceSystem,
    node_totals: Mapping[int, float],
    mass_balance_rel_sigma: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    x_map: np.ndarray,
    prior_enabled: bool = False,
    prior_mean: np.ndarray | None = None,
    prior_covariance: np.ndarray | None = None,
    prior_covariance_inflation: float = 1.0,
    aggregate_blocks: Sequence[GaussianLinearBlock] = (),
    allocation_blocks: Sequence[GaussianLinearBlock] = (),
    settings: HMCSamplerSettings | None = None,
) -> HMCSamplerResult:
    """Assemble and sample the standard flow-based MFA posterior."""

    blocks = build_flow_blocks(
        n_flows=n_flows,
        observed_flow_idx=observed_flow_idx,
        observed_values=observed_values,
        observed_sigmas=observed_sigmas,
        mass_balance=mass_balance,
        node_totals=node_totals,
        mass_balance_rel_sigma=mass_balance_rel_sigma,
        aggregate_blocks=aggregate_blocks,
        allocation_blocks=allocation_blocks,
    )

    prior = None

    if prior_enabled:
        if prior_mean is None or prior_covariance is None:
            raise ValueError(
                "prior_enabled=True requires both prior_mean and "
                "prior_covariance"
            )

        prior = TransferredGaussianPrior(
            mean=prior_mean,
            covariance=prior_covariance,
            covariance_inflation=prior_covariance_inflation,
        )

    return sample_hmc_nuts(
        lower=lower_bounds,
        upper=upper_bounds,
        blocks=blocks,
        x_map=x_map,
        prior=prior,
        settings=settings,
    )


def save_hmc_nuts_result(
    result: HMCSamplerResult,
    *,
    samples_path: str | Path,
    trace_path: str | Path,
    summary_path: str | Path,
    diagnostics_path: str | Path | None = None,
    posterior_mean_path: str | Path | None = None,
) -> None:
    """Save samples, ArviZ trace, convergence summary, and diagnostics."""

    samples_path = Path(samples_path)
    trace_path = Path(trace_path)
    summary_path = Path(summary_path)

    output_paths = [
        samples_path,
        trace_path,
        summary_path,
    ]

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        output_paths.append(diagnostics_path)

    if posterior_mean_path is not None:
        posterior_mean_path = Path(posterior_mean_path)
        output_paths.append(posterior_mean_path)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.save(samples_path, result.samples)
    result.inference_data.to_netcdf(trace_path)
    result.summary.to_csv(summary_path)

    if diagnostics_path is not None:
        with diagnostics_path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                result.diagnostics,
                stream,
                indent=2,
                sort_keys=True,
            )

    if posterior_mean_path is not None:
        import pandas as pd

        posterior_mean = result.samples.mean(axis=(0, 1))

        pd.DataFrame({
            "flow_idx": np.arange(
                posterior_mean.size,
                dtype=int,
            ),
            "posterior_mean": posterior_mean,
        }).to_csv(
            posterior_mean_path,
            index=False,
        )
