# Model formulation

The network has **234 flows** and **105 nodes**, of which **62 are internal**
(they have at least one inflow and at least one outflow, and therefore carry a
mass-balance constraint).

## Flow-based formulation

Decision variables `x ∈ R^n` (n = 234) are the flow magnitudes themselves, in
linear space.

**Objective** (quadratic):

```
  0.5 (x − μ)ᵀ Σ_prior⁻¹ (x − μ)              [prior term, target years only]
+ 0.5 Σ_j Σ_s (x_j − y_sj)² / σ_sj²            [observation term, all sources s]
+ 0.5 Σ_a (c_aᵀ x − v_a)² / σ_a²               [aggregate observations]
+ 0.5 (A_balance x)ᵀ W_mb (A_balance x)        [mass balance, soft mode]
+ 0.5 (R x)ᵀ W (R x)                            [allocation term, target years only]
```

**Multi-source likelihood.** A flow observed by several sources contributes one
term per source rather than one term per flow, so disagreeing sources pull the
posterior between them in inverse proportion to their variances. A flow with a
single observation reduces exactly to the previous single-term form, and an
unobserved flow contributes nothing. Sources and their pedigree-derived sigmas
are listed per observation in `source_inventory.csv`.

**Aggregate observations.** Some sources publish a total spanning several model
flows. Each is entered as one row `c_aᵀ x = v_a` constraining the *sum*, which
states exactly what the source reports; splitting the total across flows in
proportion to another source would invent detail the source never published and
partly restate the source being compared against. Reported against reconciled
values in `aggregate_observations.csv`.

**Mass balance** (`mass_balance.mode`):

- `soft` (default): a penalty with `σ_i = mb_rel_sigma × T_i`, where `T_i` is
  node `i`'s throughput from a MAP warm-up. Residuals are then finite and
  interpretable — reported in both kt/y and as a percentage of node throughput —
  rather than driven to machine epsilon. This is what R2-5 asks for: the data
  are allowed to disagree with closure, and the size of that disagreement is a
  result rather than an artifact.
- `hard`: `A_balance @ x = 0` as an equality constraint, one row per internal
  node (62 rows). Retained as a comparison case; residuals reach ~1e-13.

**Bounds** (hard in both modes): `x_lb ≤ x ≤ x_ub`, with `x_lb = 0.2·y_obs` and
`x_ub = 1.8·y_obs` for observed flows, and `x_lb = 0`, `x_ub = 1.8·max(y_obs)`
for unobserved flows. Where an observation is exactly zero the upper bound is
widened by `1e-6` rather than the lower bound lowered, so flows remain
non-negative.

**MAP solve:** `scipy.optimize.minimize(method="trust-constr")` with `Bounds`
for the box, plus `LinearConstraint` for the equalities in `hard` mode. The
objective is exactly quadratic, so the constant Hessian `Q` is supplied
analytically; without it the solver falls back to a BFGS approximation and does
not converge within a practical iteration budget.

## Posterior sampling

The posterior is Gaussian with precision `Q` (the MAP Hessian), restricted to
the box and, in `hard` mass-balance mode, to the mass-balance subspace.

In `soft` mode the penalty is already inside `Q`, so the sampler is handed a
zero-row `A_balance` and works in the full 234-dimensional box; passing the real
`A_balance` as well would re-impose closure as a hard constraint and double-count
it. Sampling dimension therefore rises from 172 to 234, costing roughly 35% more
time per fit.

**`truncated_gibbs_sample_nullspace` (default).** Writing `x = x_map + W u`,
where the columns of `W` span the nullspace of `A_balance` and are whitened so
that `u` has precision `I`, mass balance holds by construction for any `u` and
the target becomes a standard normal truncated to a polytope. Each sweep updates
every coordinate of `u` from its univariate truncated conditional, and then
performs a number of hit-and-run moves along random directions: in whitened
coordinates the conditional along any unit direction `d` is `N(−u·d, 1)`
truncated to the feasible segment. Coordinate moves alone mix poorly in the
corners of the polytope, so both kernels are used; each leaves the same target
invariant.

Every draw satisfies both constraint sets exactly. Draws are correlated, so
`n_tune` sweeps are discarded and `thin` is available.

**`laplace_sample_nullspace` (legacy).** Draws from the unconstrained Laplace
approximation, clips to the bounds, then projects onto the mass-balance
subspace. Because the projection runs last it can move a draw back outside the
box, so the returned draws are not guaranteed to be feasible. Retained only to
reproduce results from the original notebooks; it warns when selected.

**Convergence.** Split-R̂ and effective sample size are computed per flow on
every run (`diagnostics.py`) and written to
`convergence_summary.csv`. Flows failing R̂ ≤ 1.05 or ESS ≥ 400 are
flagged in a warning.

## 2017 → 2022 transfer

Two distinct pieces of information are carried forward. They derive from the
same 2017 posterior and are therefore **not independent**.

**Posterior as prior** (`transfer.py`). The 2017 draws are summarized into a
mean and covariance, with a configurable inflation factor (default 1.5×) so the
transferred prior is not overconfident about a five-year-old state.

Because the 2017 draws lie on the mass-balance nullspace, that covariance is
singular in the constrained directions. `precision_from_cov` inverts it by
eigendecomposition with a relative floor: directions below the threshold receive
zero precision. A direct `np.linalg.inv` there returns an indefinite matrix and
makes the objective non-convex. Note also that a sample covariance needs many
more draws than dimensions to be stable, so the 2017 draw count must stay well
above the ~160 estimated dimensions.

**Allocation ratios as soft constraints** (`allocation.py`). For a node `p` with
outflow set `O(p)` and 2017 share `s_j`, the ratio constraint is exactly linear
because `s_j` is a constant:

```
x_j / Σ_{k∈O(p)} x_k = s_j   ⟺   x_j − s_j · Σ_{k∈O(p)} x_k = 0
```

giving `R[j,j] = 1 − s_j` and `R[j,k] = −s_j` for `k ≠ j`. Each node's rows sum
to zero, so a node of degree `d` contributes rank `d−1`. Applied as a penalty
`0.5 (R x)ᵀ W (R x)`, i.e. `Q += Rᵀ W R`, which is positive semi-definite, so
`Q` stays PSD and the information reaches the sampler unchanged.

The tolerance is

```
τ_j = clip(relaxation × σ_2017,j, min_share_sigma, max_share_sigma) × T_ref,p
```

where `T_ref,p` is the node's total outflow taken from a MAP warm-up in the
target year, so the linearization scale is self-consistent with that year.
`relaxation` is the adjustable strength; sweep it with
`scripts/run_allocation_sensitivity.py` before reporting.

`allocation_residuals.csv` reports the realized deviation per row in
share units. Large values indicate that the target year's own data disagrees
with the transferred split, which is a substantive finding rather than an error.

## Node-allocation formulation

A second, independent parameterisation of the same reconciliation. Run it with
`arcade-mfa run-node-allocation --config configs/aluminum_2017_node_allocation.yaml`.

**Variables.** Each source node `p` carries a throughput `T_p` and the shares
`s_{p,j}` that split it across its outflows, with flows recovered as

```
x_j = T_p * s_{p,j},   p = from_node(j),   sum_{j in O(p)} s_{p,j} = 1
```

80 node totals plus 234 shares under 80 simplex equalities leaves 234 degrees of
freedom -- the same as the flow count, as a reparameterisation must.

**Why it exists.** It is the formulation SI Equation S17 belongs to. The
pedigree score sets a Dirichlet concentration on each node's split,

```
kappa_p = kappa_min + q_p (kappa_max - kappa_min)
```

which is a statement about *allocation* that the flow formulation cannot express
directly. `q_p` is the unweighted mean of the quality indices of the node's
outflows, each recovered by inverting S15 from the relative sigma the likelihood
actually applies -- so a customs-tracked flow is correctly treated as well
evidenced.

**Converting the data.** Per source node:

| observed outflows | contributes |
|---|---|
| all | normal likelihood on the node total (sigmas in quadrature) and a Dirichlet on its split |
| some | the per-flow normal terms are retained, so no observation is discarded |
| none | mass balance and the transferred prior only |

For 2017 this maps 79 of 80 nodes fully, giving 199 Dirichlet entries. For 2022
only 15 map fully, so most nodes rely on the Dirichlet prior carried from 2017 --
which is what the SI describes.

**Mode parameterisation.** The Dirichlet is written `alpha_j = 1 + kappa_p
shat_j`. This is not cosmetic: under the bare `alpha_j = kappa_p shat_j` the
density is unbounded as `s_j -> 0` wherever `alpha_j < 1`, so the MAP is
degenerate at the simplex boundary. Adding one places the mode exactly at the
observed share and keeps each term convex in `log s_j`. `kappa` then reads as
concentration in excess of uniform.

**Solution.** The objective is not quadratic -- mass balance is bilinear in
`(T, s)` and the Dirichlet terms are logarithmic -- so `solve_map` does not
apply. It is solved unconstrained in a transform of exactly `n_flows`
dimensions: `log` for the totals, additive log-ratio within each simplex. Both
constraints then hold by construction rather than to a solver tolerance, and the
small shares sit on a log scale. Solving in the constrained space instead is
badly conditioned: shares span four orders of magnitude and the Dirichlet
curvature reaches ~1e6, which drives a bounded equality-constrained solver into
a large excursion it then has to crawl back from.

Convergence is reported as `max |dF/dz|` against `1e-4 * |objective|`, because
an absolute gradient tolerance is unreachable on an objective of order 1e3.

**Posterior.** A Laplace approximation at the mode of the transformed density,
including the change-of-variables Jacobian, pushed back through the inverse
transform. This is weaker than the flow formulation's truncated Gibbs and is
intended for cross-checking; `tests/test_node_allocation.py` bounds it by
recovering an analytically known Dirichlet posterior. Directions that are flat
at the mode are capped and warned about: those quantities are not identified by
the data and their spread must not be quoted as a credible interval.

Mass balance is soft only. Hard closure would need a nonlinear equality
constraint, which is not implemented; the configuration raises rather than
silently ignoring `mass_balance.mode: hard`.

### Agreement between the formulations

`scripts/compare_formulations.py` writes a per-flow comparison to
`runs/si_tables/formulation_comparison.csv`.

| | 2017 | 2022 |
|---|---|---|
| flows at or above 100 kt/y | 133 | 131 |
| median absolute relative difference | 8.1% | 6.9% |
| within 25% | 107/133 | 123/131 |
| **credible intervals overlap** | **132/133** | **130/131** |
| median CI width, flow-based | 281 kt | 211 kt |
| median CI width, node-allocation | 371 kt | 262 kt |

The two formulations agree within their own stated uncertainty on all but one
flow in each year, while differing by several percent in the point estimates.
That is the expected signature of a genuine cross-check: the likelihoods differ
-- independent normals per flow against node totals plus Dirichlet splits -- so
the estimates should not coincide exactly, and the fact that the intervals still
overlap is evidence the reconciliation is driven by the data and the network
rather than by the choice of parameterisation.

## Identifiability

Only about a fifth of 2022 flows carry an observation. Some of the remainder are
weakly identified: their posterior spread approaches that of a uniform
distribution over the feasible range, and flows tied together by a shared
mass-balance constraint can be individually undetermined even when their sum is
well constrained. Where this occurs the interval reflects the bound heuristic
rather than evidence, and should be reported as such. The allocation constraints
exist to supply that missing structure; the sensitivity sweep quantifies how
much of the result depends on them.
