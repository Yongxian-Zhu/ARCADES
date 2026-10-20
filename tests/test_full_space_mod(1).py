import numpy as np
import arviz as az
import sys
import os
import multiprocessing as mp

# 1. Get the 'spawn' context
mp_ctx = mp.get_context("spawn")
# Try limiting the number of threads for the linear algebra library
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa

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

samples = rmfa.full_space_sampling(mu, np.diag(variance))
print(az.summary(samples))
#mean_posterior = az.summary(samples, var_names=["mu_x"])["mean"].to_frame()
#mean_posterior["MAP_QP"] = mu
#mean_posterior.to_csv("test_full_space_result.csv")
x_posterior = az.summary(samples, var_names=["mu_x"])["mean"].to_numpy()
rmfa.write_result_csv(x_posterior, "full_space_res")
resid = rmfa.balance_residuals(x_posterior)


