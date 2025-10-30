import numpy as np
import arviz as az
import sys
from scipy.optimize import linprog
from typing import Tuple

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa, prior_type

f_score = "/Users/dthierry/Projects/tue-27-may-25/input_file_score.csv"
f_input = "/Users/dthierry/Projects/tue-27-may-25/input_file.csv"

rmfa = ReducedSpaceMfa()
rmfa.pre_process_graph(f_input)
rmfa.read_data(3, 1, has_scores=True, score_file=f_score)

rmfa.run_parametric_optimization(is_q_param=False, has_data=True)

resv = rmfa.process_opt_result_vector()
A = rmfa.create_mfa_matrix()

rmfa.reduced_space_matrices()

Zpseudo = np.linalg.pinv(rmfa.Z)

# let mu = resv then we can have that the mu_z = Zpseudo @ mu
# this is true because rhs = 0
mu = resv

mu_z = Zpseudo @ mu

# quadratic form affine transformation of sigma

# node supply/demand bound
sup_dem_bound = rmfa.data_matrix.max() * 1.8

mu_0, variance = rmfa.simple_data_moments2()
x_lb = mu_0 * 0.2
x_lb[x_lb > mu] = mu[x_lb > mu] - 1e-3

x_ub = mu_0 * 1.8
x_ub[x_ub <= 1e-03] = sup_dem_bound

n_arc = rmfa.n_arc
x_lb[n_arc:] = -sup_dem_bound
x_ub[n_arc:] = sup_dem_bound



def find_uniform_z_bounds(a: np.ndarray, b: np.ndarray, c: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the tightest bounding box [a_z, b_z] for the uniform distribution of z.

    The problem is solved using Linear Programming (LP) for each component z_i,
    subject to the constraints derived from a <= Z*z + c <= b.

    Args:
        a: Lower bound vector for x (n-dim).
        b: Upper bound vector for x (n-dim).
        c: Translation vector (n-dim).
        Z: Transformation matrix (n x m).

    Returns:
        A tuple (a_z, b_z) containing the minimum and maximum bound vectors
        for z (m-dim). Returns NaN for bounds if the LP fails.
    """

    # 1. Determine dimensions
    m = Z.shape[1] # Dimension of z (number of columns in Z)

    # 2. Construct the combined constraint matrix A_ub and vector b_ub
    # Constraints: Z*z <= b - c  AND  -Z*z <= c - a

    A_ub = np.vstack([Z, -Z])
    #b_ub = np.concatenate([b - c, c - a])
    #b_ub = np.hstack([b - c, c - a])
    b_ub = np.hstack([b, -a])

    # 3. Initialize bounds storage
    a_z = np.zeros(m)
    b_z = np.zeros(m)

    # 4. Solve Linear Programs for each component z_i
    for i in range(m):

        # --- Objective Vector Setup ---

        # Minimization (to find a_z[i]): c_obj = e_i (unit vector)
        c_min = np.zeros(m)
        c_min[i] = 1.0

        # Maximization (to find b_z[i]): c_obj = -e_i (equivalent to minimizing -z_i)
        c_max = np.zeros(m)
        c_max[i] = -1.0

        # --- Solve for Minimum z_i (a_z[i]) ---
        res_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub, method='highs', bounds
                          = (None, None))
        if res_min.success:
            # The result (res_min.fun) is min(z_i)
            a_z[i] = res_min.fun
        else:
            a_z[i] = np.nan
            print(f"Warning: LP failed to find min z_{i+1}. Status: {res_min.message}")

        # --- Solve for Maximum z_i (b_z[i]) ---
        res_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub, method='highs',
                          bounds=(None, None))
        if res_max.success:
            # The result is min(-z_i), which is -max(z_i). Negate to get max(z_i).
            b_z[i] = -res_max.fun
        else:
            b_z[i] = np.nan
            print(f"Warning: LP failed to find max z_{i+1}. Status: {res_max.message}")

    return a_z, b_z

# ==============================================================================
# Example Usage
# ==============================================================================

# Input parameters from the previous example (2x2 case)

# Calculate the bounds
# x_lb = np.ones(mu_0.size) * -sup_dem_bound
# x_ub = np.ones(mu_0.size) * sup_dem_bound
a_z_bounds, b_z_bounds = find_uniform_z_bounds(x_lb,
                                               x_ub,
                                               np.zeros(x_lb.size),
                                               rmfa.Z)

print("-" * 40)
print(f"The tightest bounding box for z is defined by:")
for i in range(len(a_z_bounds)):
    print(f"  z_{i+1} is in [{a_z_bounds[i]:.4f}, {b_z_bounds[i]:.4f}]")


#
#
# x_zlb = Zpseudo @ x_lb
# x_zub = Zpseudo @ x_ub
#
# print(f"variance shape = {variance.shape}")
# print(f"Z shape = {rmfa.Z.shape}")
# print(f"Zpseudo shape = {Zpseudo.shape}")
#
variance_z = np.diag(variance) @ Zpseudo.transpose()
variance_z = Zpseudo @ variance_z
# #
# # print(variance_z)
# #
# #
# # # now we want to have a lower bound and an upper bound for the uniform
# # # distribution
# # n_var = rmfa.n_arc + len(rmfa.ds_nodes)
# #
z_samples = rmfa.red_space_mult_data_sampling(mu_z,
                                              variance_z,
                                              prior_type(2),
                                              x_zlb=a_z_bounds,
                                              x_zub=b_z_bounds
                                              )
print(az.summary(z_samples))

x_posterior = az.summary(z_samples, var_names=["mu_x"])["mean"].to_numpy()
rmfa.write_result_csv(x_posterior, "red_space_res")
resid = rmfa.balance_residuals(x_posterior)

