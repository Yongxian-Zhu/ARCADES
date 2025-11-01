import numpy as np
import arviz as az
import sys
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import os

# 1. Get the 'spawn' context
mp_ctx = mp.get_context("spawn")
# Try limiting the number of threads for the linear algebra library
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

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

# this is true because rhs = 0
mu = resv
_, variance = rmfa.simple_data_moments()

sup_dem_bound = rmfa.data_matrix.max() * 1.8
mu_0, variance = rmfa.simple_data_moments2()
x_lb = mu_0 * 0.2
x_lb[x_lb > mu] = mu[x_lb > mu] - 1e-3


x_ub = mu_0 * 1.8
x_ub[x_ub <= 1e-03] = sup_dem_bound

n_arc = rmfa.n_arc
x_lb[n_arc:] = -sup_dem_bound
x_ub[n_arc:] = sup_dem_bound

mu_x_init = mu.copy()
#mu_x_init[np.abs(mu)< 1e-03] = 1e-3

samples = rmfa.full_space_sampling(mu_x_init, np.diag(variance),
                                   prior_type(2),
                                   x_lb=x_lb,
                                   x_ub=x_ub
                                   )
print(az.summary(samples))
#mean_posterior = az.summary(samples, var_names=["mu_x"])["mean"].to_frame()
#mean_posterior["MAP_QP"] = mu
#mean_posterior.to_csv("test_full_space_result.csv")
x_posterior = az.summary(samples, var_names=["mu_x"])["mean"].to_numpy()
rmfa.write_result_csv(x_posterior, "full_space_res")
resid = rmfa.balance_residuals(x_posterior)


# save the samples
az.to_netcdf(samples, "full_space_res.nc")

mu_flat = samples.posterior["mu_x"].stack(sample=("chain", "draw")).values

# save the posterior values
np.save("full_space_res.npy", mu_flat)

# plot result folders
res_dir_path = 'full_space_res'
if not os.path.exists(res_dir_path):
    os.mkdir(res_dir_path)
else:
    print(f"The directory {res_dir_path} already exists.")

for flow_idx in range(rmfa.n_arc):
    # choose an index

    # plot histograms
    plt.figure(figsize=(7, 4))
    plt.hist(mu_flat[flow_idx, :], bins=50, density=True, alpha=0.6, color="steelblue")

    mean = mu_flat[flow_idx, :].mean()
    low, high = np.quantile(mu_flat[flow_idx, :], [0.025, 0.975])

    plt.axvline(mean, color="k", linestyle="--", label=f"mean = {mean:.3f}")
    plt.axvline(low,  color="red", linestyle=":", label=f"2.5% = {low:.3f}")
    plt.axvline(high, color="red", linestyle=":")
    plt.title("Posterior of μ")
    plt.xlabel("μ")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_dir_path + f"/hist_{flow_idx}.png")
    plt.close()

