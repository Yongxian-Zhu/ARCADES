import os
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from numpy.linalg import solve, lstsq

# -------------------------
# Configuration
# -------------------------
fn_data_2022 = "flow_data_2022.csv"   # CSV with columns: Flow index, from_node_name, to_node_name,
                                      # from_node_number, to_node_number, Value1, Value2, Value3, Value4
fn_score_data = "flow_score_2022.csv"     # File containing score data (CSV or other format)
prior_mean_path = "prior_mean_first241.npy"
prior_cov_path  = "prior_cov_first241.npy"

out_prefix = "full_space_res_2022"
out_dir    = out_prefix

# Posterior sampling config
n_chains = 4
n_draws  = 2000  # total samples = n_chains * n_draws = 8000
rng = np.random.default_rng(42)

# -------------------------
# Load priors (first 241 variables)
# -------------------------
if not (os.path.exists(prior_mean_path) and os.path.exists(prior_cov_path)):
    raise FileNotFoundError("Missing prior files: prior_mean_first241.npy and prior_cov_first241.npy")

mu_prior = np.load(prior_mean_path)        # shape (241,)
cov_prior = np.load(prior_cov_path)        # shape (241, 241)
cov_prior = cov_prior + 1e-10 * np.eye(cov_prior.shape[0])  # jitter for SPD stability
inv_cov_prior = np.linalg.inv(cov_prior)

n_vars = 241
if mu_prior.shape[0] < n_vars or cov_prior.shape[0] < n_vars:
    raise ValueError(f"Prior size mismatch. mean={mu_prior.shape[0]}, cov={cov_prior.shape}")

# -------------------------
# Load sparse 2022 data (Value1 used as observations, other columns may contain scores)
# -------------------------
df = pd.read_csv(fn_data_2022)
# Convert numeric columns
for col in ["Flow index", "from_node_number", "to_node_number", "Value1", "Value2", "Value3", "Value4"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Print available columns to understand data structure
print(f"Available columns in CSV: {list(df.columns)}")
print(f"Data shape: {df.shape}")

# Map Flow index to variable indices 0..240 (auto-convert 1-based to 0-based)
if "Flow index" not in df.columns:
    raise ValueError("flow_data_2022.csv must have 'Flow index' column to align flows to variables.")
flow_idx_raw = df["Flow index"].dropna().astype(int).to_numpy()
zero_based = (flow_idx_raw.min() == 0)
flow_idx = df["Flow index"].to_numpy()
var_idx = np.array(
    [int(i - (0 if zero_based else 1)) if not np.isnan(i) else -1 for i in flow_idx],
    dtype=int
)

# Keep only rows for variables 0..240
valid_rows = (var_idx >= 0) & (var_idx < n_vars)
df_valid = df.loc[valid_rows].copy()
var_idx_valid = var_idx[valid_rows]

# Observations: Value1 median per variable (NaN if missing)
df_valid["Value1"] = pd.to_numeric(df_valid["Value1"], errors="coerce")
y_obs = np.full(n_vars, np.nan, dtype=float)
agg_vals = (
    pd.DataFrame({"idx": var_idx_valid, "value": df_valid["Value1"].to_numpy()})
    .groupby("idx", as_index=False)["value"].median()
)
y_obs[agg_vals["idx"].to_numpy()] = agg_vals["value"].to_numpy()
obs_mask = ~np.isnan(y_obs)

# -------------------------
# Load score data from separate file
# -------------------------
scores = np.full(n_vars, np.nan, dtype=float)

# Load score data file
score_file_found = False
score_file_path = fn_score_data
if os.path.exists(score_file_path):
    print(f"Loading score data from: {score_file_path}")
    
    if score_file_path.endswith(".npy"):
        # Load numpy array
        score_data = np.load(score_file_path)
        if score_data.shape[0] >= n_vars:
            scores[:n_vars] = score_data[:n_vars]
        else:
            scores[:score_data.shape[0]] = score_data
        print(f"Loaded {len(score_data)} scores from numpy file")
        
    else:
        # Load CSV/text file
        try:
            df_scores = pd.read_csv(score_file_path)
            print(f"Score file columns: {list(df_scores.columns)}")
            
            # Try to find Flow index and score columns
            flow_idx_col = None
            score_col = None
            
            # Look for flow index column
            for col in ["Flow index", "flow_index", "Flow_index", "index", "Index"]:
                if col in df_scores.columns:
                    flow_idx_col = col
                    break
            
            # Look for score column - check for numeric columns that could contain scores
            numeric_cols = []
            for col in ["Score", "score", "Value", "value", "Rating", "rating", "Value1", "Value2", "Value3", "Value4"]:
                if col in df_scores.columns:
                    try:
                        # Test if column can be converted to numeric
                        pd.to_numeric(df_scores[col].dropna().iloc[:5], errors='raise')
                        numeric_cols.append(col)
                    except (ValueError, IndexError):
                        continue
            
            if flow_idx_col is None:
                # Assume first column is flow index
                flow_idx_col = df_scores.columns[0]
                print(f"Assuming first column '{flow_idx_col}' is flow index")
            
            if len(numeric_cols) > 0:
                # Use the first numeric column found as scores
                score_col = numeric_cols[0]
                print(f"Using column '{score_col}' as scores (first numeric column found)")
            else:
                # Fallback: assume last column contains scores
                score_col = df_scores.columns[-1]
                print(f"No obvious score column found, assuming last column '{score_col}' contains scores")
            
            # Convert to numeric
            df_scores[flow_idx_col] = pd.to_numeric(df_scores[flow_idx_col], errors="coerce")
            df_scores[score_col] = pd.to_numeric(df_scores[score_col], errors="coerce")
            
            # Remove rows with NaN values
            df_scores_clean = df_scores.dropna(subset=[flow_idx_col, score_col])
            
            # Convert flow indices to 0-based if needed
            flow_indices = df_scores_clean[flow_idx_col].astype(int).to_numpy()
            score_values = df_scores_clean[score_col].to_numpy()
            
            # Check if indices are 1-based or 0-based
            if len(flow_indices) > 0 and flow_indices.min() == 1:
                flow_indices = flow_indices - 1  # Convert to 0-based
                print("Converted 1-based flow indices to 0-based")
            
            # Assign scores to variables
            valid_indices = (flow_indices >= 0) & (flow_indices < n_vars)
            scores[flow_indices[valid_indices]] = score_values[valid_indices]
            
            print(f"Loaded scores for {np.sum(valid_indices)} flows")
            
        except Exception as e:
            print(f"Error reading score file {score_file_path}: {e}")
        
    score_file_found = True
else:
    print(f"Warning: Score file '{fn_score_data}' not found")
    print("All flows will be treated as non-score-1 flows (using default bounds)")

# Report score statistics
score_counts = pd.Series(scores[~np.isnan(scores)]).value_counts().sort_index()
print(f"Score distribution:")
for score_val, count in score_counts.items():
    print(f"  Score {score_val:.0f}: {count} flows")

# Create mask for flows with score = 1
score_1_mask = (scores == 1.0) & obs_mask  # Only apply to observed flows with score 1
print(f"Found {np.sum(score_1_mask)} flows with score 1 (out of {np.sum(obs_mask)} observed flows)")

# -------------------------
# Build mass-balance constraints using node appearance in both columns
# -------------------------
# Identify internal nodes: nodes that appear in both from_node_number and to_node_number
from_nodes_all = df["from_node_number"].dropna().astype(int).to_numpy()
to_nodes_all   = df["to_node_number"].dropna().astype(int).to_numpy()
internal_nodes = set(from_nodes_all).intersection(set(to_nodes_all))

# Now restrict to the first 241 variables (df_valid) to build constraint rows
from_valid = df_valid["from_node_number"].to_numpy()
to_valid   = df_valid["to_node_number"].to_numpy()

A_rows = []
for node in sorted(internal_nodes):
    # Collect variable indices for inflows/outflows for this node within the first 241 variables
    inflow_vars = var_idx_valid[(to_valid == node)]
    outflow_vars = var_idx_valid[(from_valid == node)]
    if len(inflow_vars) == 0 and len(outflow_vars) == 0:
        # Node is internal globally but has no flows among the first 241 variables; skip
        continue
    row = np.zeros(n_vars, dtype=float)
    for j in inflow_vars:
        row[j] += 1.0
    for j in outflow_vars:
        row[j] -= 1.0
    # If node has both inflows and outflows (within subset), enforce mass balance
    if (len(inflow_vars) > 0) and (len(outflow_vars) > 0):
        A_rows.append(row)
    # If only inflows or only outflows within subset, treat as source/sink for this subset (no constraint)

A_bal = np.vstack(A_rows) if A_rows else np.zeros((0, n_vars), dtype=float)
b_bal = np.zeros(A_bal.shape[0], dtype=float)
print(f"Mass-balance: constraints for {A_bal.shape[0]} internal nodes (subset of first 241 variables).")

# -------------------------
# Bounds and observational uncertainty - NON-NEGATIVE FLOWS
# -------------------------
max_obs = np.nanmax(y_obs) if np.any(obs_mask) else 1.0
sup_dem_bound = float(max(1.0, max_obs * 1.8))

# PHYSICAL CONSTRAINT: All flows must be non-negative (>= 0)
# Allow zero flows for variables that should be zero

# Set bounds based on score values
# For flows with score 1: ±10% bounds (lower bound = 0.9 * obs, upper bound = 1.1 * obs)
# For other flows: ±30% bounds (lower bound = 0.7 * obs, upper bound = 1.3 * obs)

# Initialize lower bounds
x_lb = np.zeros(n_vars, dtype=float)  # Default: all flows >= 0

# For flows with score 1, set lower bound to 90% of observation
# For other observed flows, set lower bound to 70% of observation
x_lb = np.where(score_1_mask, 
                0.9 * y_obs,  # 90% of observed value for score 1 flows
                np.where(obs_mask, 0.7 * y_obs, 0.0))  # 70% for other observed, 0 for unobserved

# Initialize upper bounds
x_ub = np.where(obs_mask, 
                1.3 * y_obs,  # Default: 130% of observed value (±30%)
                sup_dem_bound)  # Default upper bound for unobserved

# For flows with score 1, set upper bound to 110% of observation
x_ub = np.where(score_1_mask,
                1.1 * y_obs,  # 110% of observed value for score 1 flows (±10%)
                x_ub)         # Keep default bounds for other flows

# For observed flows that are very small (< 1e-3), ensure minimum bounds
small_flow_threshold = 1e-3
small_flows = obs_mask & (y_obs < small_flow_threshold)
if np.any(small_flows):
    print(f"Found {np.sum(small_flows)} observed flows < {small_flow_threshold} - adjusting bounds")
    # For score 1 flows that are small
    small_score1 = small_flows & score_1_mask
    if np.any(small_score1):
        x_lb[small_score1] = np.maximum(0.9 * y_obs[small_score1], 1e-4)  # Minimum lower bound
        x_ub[small_score1] = np.maximum(1.1 * y_obs[small_score1], 1e-3)  # Minimum upper bound
    
    # For other small flows (not score 1)
    small_other = small_flows & ~score_1_mask
    if np.any(small_other):
        x_ub[small_other] = np.maximum(1.3 * y_obs[small_other], 0.1)

print(f"Bounds configuration:")
print(f"- Score 1 flows: ±10% bounds (90%-110% of observation)")
print(f"- Other observed flows: ±30% bounds (70%-130% of observation)")
print(f"- Unobserved flows: non-negative with upper bound = {sup_dem_bound:.1f}")
print(f"Lower bounds range: [{x_lb.min():.6f}, {x_lb.max():.6f}]")
print(f"Upper bounds range: [{x_ub.min():.6f}, {x_ub.max():.6f}]")

if np.any(score_1_mask):
    score1_lb_range = x_lb[score_1_mask]
    score1_ub_range = x_ub[score_1_mask]
    score1_obs_range = y_obs[score_1_mask]
    print(f"Score 1 flows ({np.sum(score_1_mask)} flows):")
    print(f"  Observations range: [{score1_obs_range.min():.6f}, {score1_obs_range.max():.6f}]")
    print(f"  Lower bounds range: [{score1_lb_range.min():.6f}, {score1_lb_range.max():.6f}]")
    print(f"  Upper bounds range: [{score1_ub_range.min():.6f}, {score1_ub_range.max():.6f}]")

if np.any(obs_mask & ~score_1_mask):
    other_obs_mask = obs_mask & ~score_1_mask
    other_ub_range = x_ub[other_obs_mask]
    other_obs_range = y_obs[other_obs_mask]
    other_lb_range = x_lb[other_obs_mask]
    print(f"Other observed flows ({np.sum(other_obs_mask)} flows):")
    print(f"  Observations range: [{other_obs_range.min():.6f}, {other_obs_range.max():.6f}]")
    print(f"  Lower bounds range: [{other_lb_range.min():.6f}, {other_lb_range.max():.6f}]")
    print(f"  Upper bounds range: [{other_ub_range.min():.6f}, {other_ub_range.max():.6f}]")

# Observational uncertainty
sigma_obs = np.where(obs_mask, np.maximum(0.10 * np.abs(y_obs), 1e-3), np.inf)
inv_obs_var = np.where(np.isfinite(sigma_obs), 1.0 / np.maximum(sigma_obs**2, 1e-12), 0.0)

# -------------------------
# Quadratic MAP objective
# -------------------------
Q = inv_cov_prior + np.diag(inv_obs_var)
q = -inv_cov_prior @ mu_prior - (inv_obs_var * np.nan_to_num(y_obs, nan=0.0))

def obj_fun(x):
    return 0.5 * float(x @ (Q @ x)) + float(q @ x)

def obj_grad(x):
    return Q @ x + q

# -------------------------
# Initial point: project prior onto mass-balance constraints, then clip to non-negative
# -------------------------
x0 = mu_prior.copy()
if A_bal.shape[0] > 0:
    AMu = A_bal @ mu_prior
    AAT = A_bal @ A_bal.T
    try:
        lam = solve(AAT + 1e-10 * np.eye(AAT.shape[0]), AMu)
    except np.linalg.LinAlgError:
        lam = lstsq(AAT + 1e-10 * np.eye(AAT.shape[0]), AMu, rcond=None)[0]
    x0 = mu_prior - A_bal.T @ lam

# CRITICAL: Ensure initial point respects bounds (including lower bounds for score 1 flows)
x0 = np.clip(x0, x_lb, x_ub)

# Check if any initial values violate lower bounds
below_lb_count = np.sum(x0 < x_lb)
if below_lb_count > 0:
    print(f"Warning: {below_lb_count} initial values were below lower bounds, adjusting")
    x0 = np.maximum(x0, x_lb)

print(f"Initial point range: [{x0.min():.6f}, {x0.max():.6f}]")
print(f"Initial point: {np.sum(x0 == 0)} variables at zero, {np.sum(x0 > 0)} variables positive")

# -------------------------
# Solve constrained MAP
# -------------------------
lin_con = LinearConstraint(A_bal, lb=b_bal, ub=b_bal) if A_bal.shape[0] > 0 else None
bounds = Bounds(x_lb, x_ub)

res = minimize(
    fun=obj_fun,
    x0=x0,
    jac=obj_grad,
    method="trust-constr",
    bounds=bounds,
    constraints=([lin_con] if lin_con is not None else []),
    options={"maxiter": 2000, "verbose": 1}
)

if not res.success:
    print("Warning: optimizer did not fully converge:", res.message)

x_map = res.x

# CRITICAL: Verify MAP estimate respects bounds
below_lb_map = np.sum(x_map < x_lb)
above_ub_map = np.sum(x_map > x_ub)
if below_lb_map > 0 or above_ub_map > 0:
    print(f"Warning: {below_lb_map} MAP estimates below lower bounds, {above_ub_map} above upper bounds")
    print(f"MAP range before correction: [{x_map.min():.6f}, {x_map.max():.6f}]")
    # Force within bounds
    x_map = np.clip(x_map, x_lb, x_ub)
    print(f"Corrected MAP range: [{x_map.min():.6f}, {x_map.max():.6f}]")
else:
    print(f"MAP estimate range: [{x_map.min():.6f}, {x_map.max():.6f}] - All within bounds ✓")

# Check score 1 flows specifically
if np.any(score_1_mask):
    score1_map = x_map[score_1_mask]
    score1_obs = y_obs[score_1_mask]
    score1_deviations = np.abs(score1_map - score1_obs) / score1_obs * 100
    print(f"Score 1 flows MAP deviations: mean={np.mean(score1_deviations):.1f}%, max={np.max(score1_deviations):.1f}%")

print(f"MAP estimate: {np.sum(x_map == 0)} variables at zero, {np.sum(x_map > 0)} variables positive")

mb_residuals = A_bal @ x_map if A_bal.shape[0] > 0 else np.array([])
if mb_residuals.size > 0:
    print("Mass-balance residuals: min", float(mb_residuals.min()), "max", float(mb_residuals.max()))

# -------------------------
# Posterior sampling in mass-balance subspace (Laplace) - ENSURING NON-NEGATIVITY
# -------------------------
def nullspace(A, rtol=1e-10):
    if A.size == 0:
        return np.eye(n_vars)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = (s > rtol * s.max()).sum()
    Z = Vt[rank:].T
    return Z

Z = nullspace(A_bal)
Q_eff = Z.T @ Q @ Z
try:
    L_eff = np.linalg.cholesky(Q_eff + 1e-12 * np.eye(Q_eff.shape[0]))
    chol_like = True
except np.linalg.LinAlgError:
    chol_like = False
    w, V = np.linalg.eigh(Q_eff)
    w = np.maximum(w, 1e-12)
    L_eff = V @ np.diag(np.sqrt(w))

def project_to_mass_balance(x):
    if A_bal.shape[0] == 0:
        return x
    rhs = A_bal @ x
    AAT = A_bal @ A_bal.T
    try:
        lam = solve(AAT + 1e-10 * np.eye(AAT.shape[0]), rhs)
    except np.linalg.LinAlgError:
        lam = lstsq(AAT + 1e-10 * np.eye(AAT.shape[0]), rhs, rcond=None)[0]
    return x - A_bal.T @ lam

# Improved posterior sampling: use truncated normal around MAP estimate
# This ensures samples stay close to MAP while respecting constraints
samples_arr = np.empty((n_chains, n_draws, n_vars), dtype=float)
rejected_samples = 0

# Compute posterior covariance in the constrained space
if A_bal.shape[0] > 0:
    # Effective covariance in nullspace
    try:
        cov_eff_inv = Q_eff + 1e-12 * np.eye(Q_eff.shape[0])
        cov_eff = np.linalg.inv(cov_eff_inv)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Q_eff)
        w_inv = 1.0 / np.maximum(w, 1e-12)
        cov_eff = V @ np.diag(w_inv) @ V.T
else:
    # No constraints - use full space
    try:
        cov_eff = np.linalg.inv(Q + 1e-12 * np.eye(Q.shape[0]))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Q)
        w_inv = 1.0 / np.maximum(w, 1e-12)
        cov_eff = V @ np.diag(w_inv) @ V.T

# Adaptive sampling with smaller perturbations
for ci in range(n_chains):
    for di in range(n_draws):
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            if A_bal.shape[0] > 0:
                # Sample in nullspace with smaller variance
                scale_factor = 0.1 if attempts < 5 else 0.05  # Reduce scale after failures
                z0 = rng.multivariate_normal(np.zeros(Q_eff.shape[0]), scale_factor * cov_eff)
                x_s = x_map + Z @ z0
            else:
                # Sample in full space with smaller variance
                scale_factor = 0.1 if attempts < 5 else 0.05
                noise = rng.multivariate_normal(np.zeros(n_vars), scale_factor * cov_eff)
                x_s = x_map + noise
            
            # Project to satisfy mass balance constraints
            x_s = project_to_mass_balance(x_s)
            
            # Soft clipping: gradually move towards bounds rather than hard clipping
            # This preserves the structure better
            x_s_clipped = np.clip(x_s, x_lb, x_ub)
            
            # Check if clipping was minimal (good sample)
            clipping_error = np.mean(np.abs(x_s - x_s_clipped))
            if clipping_error < 0.01 * np.mean(np.abs(x_map)):
                samples_arr[ci, di, :] = x_s_clipped
                break
            else:
                attempts += 1
        
        if attempts >= max_attempts:
            # Conservative fallback: MAP + very small Gaussian noise
            # Use posterior standard deviations as scale
            if A_bal.shape[0] > 0:
                posterior_std = np.sqrt(np.diag(Z @ cov_eff @ Z.T))
            else:
                posterior_std = np.sqrt(np.diag(cov_eff))
            
            # Scale noise to be 5% of posterior std or 1% of MAP value
            noise_scale = np.minimum(0.05 * posterior_std, 0.01 * np.abs(x_map))
            noise_scale = np.maximum(noise_scale, 1e-6)  # Minimum noise
            
            noise = rng.normal(0, noise_scale)
            x_s = x_map + noise
            x_s = project_to_mass_balance(x_s)
            x_s = np.clip(x_s, x_lb, x_ub)
            samples_arr[ci, di, :] = x_s
            rejected_samples += 1

if rejected_samples > 0:
    print(f"Warning: {rejected_samples} samples used fallback sampling (conservative noise)")

# Verify all samples respect bounds
min_sample = np.min(samples_arr)
max_sample = np.max(samples_arr)
below_lb_samples = np.sum(samples_arr < x_lb[np.newaxis, np.newaxis, :])
above_ub_samples = np.sum(samples_arr > x_ub[np.newaxis, np.newaxis, :])

if below_lb_samples > 0 or above_ub_samples > 0:
    print(f"Warning: {below_lb_samples} samples below lower bounds, {above_ub_samples} above upper bounds")
    print(f"Sample range: [{min_sample:.6f}, {max_sample:.6f}]")
else:
    print(f"All samples respect bounds ✓ (range: [{min_sample:.6f}, {max_sample:.6f}])")

# Count samples by type
zero_samples = np.sum(samples_arr == 0.0)
total_samples = samples_arr.size
print(f"Samples: {zero_samples} zeros out of {total_samples} total ({100*zero_samples/total_samples:.1f}%)")

# For score 1 flows, check if samples stay within ±10%
if np.any(score_1_mask):
    score1_samples = samples_arr[:, :, score_1_mask]
    score1_obs_vals = y_obs[score_1_mask]
    score1_within_10pct = np.sum(
        (score1_samples >= 0.9 * score1_obs_vals[np.newaxis, np.newaxis, :]) &
        (score1_samples <= 1.1 * score1_obs_vals[np.newaxis, np.newaxis, :])
    )
    score1_total = score1_samples.size
    print(f"Score 1 samples within ±10%: {score1_within_10pct}/{score1_total} ({100*score1_within_10pct/score1_total:.1f}%)")

# Diagnostic: Check sample quality
sample_means = np.mean(samples_arr, axis=(0, 1))
sample_stds = np.std(samples_arr, axis=(0, 1))

# Compare with MAP and prior
map_diff = np.abs(sample_means - x_map)
prior_diff = np.abs(sample_means - mu_prior)
obs_diff = np.abs(sample_means[obs_mask] - y_obs[obs_mask]) if np.any(obs_mask) else np.array([])

print(f"\nSample Quality Diagnostics:")
print(f"Mean deviation from MAP: {np.mean(map_diff):.6f} (max: {np.max(map_diff):.6f})")
print(f"Mean deviation from prior: {np.mean(prior_diff):.6f} (max: {np.max(prior_diff):.6f})")
if len(obs_diff) > 0:
    print(f"Mean deviation from observations: {np.mean(obs_diff):.6f} (max: {np.max(obs_diff):.6f})")
print(f"Sample std range: [{np.min(sample_stds):.6f}, {np.max(sample_stds):.6f}]")

# Check if samples are too close to MAP (indicating poor mixing)
very_close_to_map = np.sum(map_diff < 1e-6)
if very_close_to_map > 0.5 * n_vars:
    print(f"WARNING: {very_close_to_map} variables have samples very close to MAP - may indicate poor mixing")

# -------------------------
# Summaries and outputs
# -------------------------
coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws), "mu_x_dim_0": np.arange(n_vars)}
posterior_ds = xr.Dataset({"mu_x": (("chain", "draw", "mu_x_dim_0"), samples_arr)}, coords=coords)
idata = az.InferenceData(posterior=posterior_ds)

summary = az.summary(idata, var_names=["mu_x"])
print(summary)
x_posterior = summary["mean"].to_numpy()

# Final verification and correction
below_lb_posterior = np.sum(x_posterior < x_lb)
above_ub_posterior = np.sum(x_posterior > x_ub)
if below_lb_posterior > 0 or above_ub_posterior > 0:
    print(f"WARNING: {below_lb_posterior} posterior means below lower bounds, {above_ub_posterior} above upper bounds")
    print(f"Posterior mean range before correction: [{x_posterior.min():.6f}, {x_posterior.max():.6f}]")
    
    # Force posterior means within bounds
    x_posterior = np.clip(x_posterior, x_lb, x_ub)
    print(f"Corrected posterior mean range: [{x_posterior.min():.6f}, {x_posterior.max():.6f}]")
    print("All posterior means now respect bounds ✓")
else:
    print(f"All posterior means respect bounds ✓ (range: [{x_posterior.min():.6f}, {x_posterior.max():.6f}])")

# Count posterior means by type
zero_posterior = np.sum(x_posterior == 0.0)
print(f"Posterior means: {zero_posterior} at zero, {n_vars - zero_posterior} positive")

# Check score 1 flows specifically
if np.any(score_1_mask):
    score1_posterior = x_posterior[score_1_mask]
    score1_obs_vals = y_obs[score_1_mask]
    score1_post_deviations = np.abs(score1_posterior - score1_obs_vals) / score1_obs_vals * 100
    print(f"Score 1 flows posterior deviations: mean={np.mean(score1_post_deviations):.1f}%, max={np.max(score1_post_deviations):.1f}%")

# Detailed comparison for observed variables
print(f"\nDetailed Comparison for Observed Variables ({np.sum(obs_mask)} total):")
if np.any(obs_mask):
    obs_indices = np.where(obs_mask)[0]
    print("Variable | Observation | Prior Mean | MAP Est. | Post. Mean | Deviation")
    print("-" * 70)
    for i in obs_indices[:10]:  # Show first 10 observed variables
        obs_val = y_obs[i]
        prior_val = mu_prior[i]
        map_val = x_map[i]
        post_val = x_posterior[i]
        deviation = abs(post_val - obs_val) / max(obs_val, 1e-6) * 100
        print(f"{i:8d} | {obs_val:11.6f} | {prior_val:10.6f} | {map_val:8.6f} | {post_val:10.6f} | {deviation:6.1f}%")
    
    if len(obs_indices) > 10:
        print(f"... and {len(obs_indices) - 10} more observed variables")
    
    # Summary statistics for observed variables
    obs_post_devs = np.abs(x_posterior[obs_mask] - y_obs[obs_mask]) / np.maximum(y_obs[obs_mask], 1e-6) * 100
    obs_map_devs = np.abs(x_map[obs_mask] - y_obs[obs_mask]) / np.maximum(y_obs[obs_mask], 1e-6) * 100
    
    print(f"\nObserved Variables Summary:")
    print(f"MAP deviations from obs:  mean={np.mean(obs_map_devs):.1f}%, max={np.max(obs_map_devs):.1f}%")
    print(f"Post deviations from obs: mean={np.mean(obs_post_devs):.1f}%, max={np.max(obs_post_devs):.1f}%")
    
    # Check if posterior is closer to observations than prior
    prior_obs_devs = np.abs(mu_prior[obs_mask] - y_obs[obs_mask]) / np.maximum(y_obs[obs_mask], 1e-6) * 100
    print(f"Prior deviations from obs: mean={np.mean(prior_obs_devs):.1f}%, max={np.max(prior_obs_devs):.1f}%")
    
    better_than_prior = np.sum(obs_post_devs < prior_obs_devs)
    print(f"Posterior is closer to observations than prior for {better_than_prior}/{len(obs_indices)} variables")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
else:
    print(f"The directory {out_dir} already exists.")

az.to_netcdf(idata, f"{out_prefix}.nc")
mu_flat = idata.posterior["mu_x"].stack(sample=("chain", "draw")).transpose("mu_x_dim_0", "sample").values
np.save(f"{out_prefix}.npy", mu_flat)

# Ensure all values in the output DataFrame are non-negative
x_posterior_safe = np.maximum(x_posterior, 0.0)
x_map_safe = np.maximum(x_map, 0.0)

posterior_mean_df = pd.DataFrame({
    "variable_idx": np.arange(n_vars),
    "posterior_mean": x_posterior_safe,
    "map_estimate": x_map_safe,
    "prior_mean": mu_prior,
    "lower_bound": x_lb,
    "upper_bound": x_ub,
    "observation": y_obs,
    "has_observation": obs_mask,
    "score": scores,
    "is_score_1": score_1_mask,
    "posterior_std": sample_stds,
    "deviation_from_obs_pct": np.where(obs_mask, 
                                       np.abs(x_posterior_safe - y_obs) / np.maximum(y_obs, 1e-6) * 100, 
                                       np.nan)
})
posterior_mean_df.to_csv(os.path.join(out_dir, f"{out_prefix}_posterior_mean.csv"), index=False)

# Generate histograms for first 10 variables to verify non-negativity
for j in range(min(10, n_vars)):
    vals = mu_flat[j, :]
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=50, density=True, alpha=0.6, color="steelblue")
    mean = vals.mean()
    low, high = np.quantile(vals, [0.025, 0.975])
    plt.axvline(mean, color="k", linestyle="--", label=f"mean = {mean:.3f}")
    plt.axvline(low,  color="red", linestyle=":", label=f"2.5% = {low:.3f}")
    plt.axvline(high, color="red", linestyle=":")
    plt.axvline(0, color="red", linestyle="-", linewidth=2, label="zero")  # Show zero line
    plt.title(f"Posterior of μ (2022) — variable {j}")
    plt.xlabel("μ")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_{j}.png"))
    plt.close()

print("Done. Bounds configuration:")
print("- Score 1 flows: ±10% bounds (90%-110% of observation)")
print("- Other observed flows: ±30% bounds (70%-130% of observation)")
print("- Unobserved flows: non-negative with default upper bound")
print("Mass balance enforced for internal nodes.")