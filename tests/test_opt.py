import pandas as pd
import numpy as np
import sys

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa, prior_type

f_score = "./data/input_file_score.csv"
f_input = "./data/input_file.csv"

rmfa = ReducedSpaceMfa()
rmfa.pre_process_graph(f_input)
rmfa.read_data(3, 1, has_scores=True, score_file=f_score)
A = rmfa.create_mfa_matrix()

rmfa.run_parametric_optimization(is_q_param=False, has_data=True)
resv = rmfa.process_opt_result_vector()

# this is true because rhs = 0
mu = resv

sup_dem_bound = rmfa.data_matrix.max() * 1.8
mu_0, variance = rmfa.simple_data_moments2()
x_lb = mu_0 * 0.2
x_lb[x_lb > mu] = mu[x_lb > mu] - 1e-3


x_ub = mu_0 * 1.8
x_ub[x_ub <= 1e-03] = sup_dem_bound

n_arc = rmfa.n_arc
x_lb[n_arc:] = -sup_dem_bound
x_ub[n_arc:] = sup_dem_bound

for i in range(n_arc):
    x_lb[i] = 0 if x_lb[i]<0 else x_lb[i]

rmfa.run_parametric_optimization(is_q_param=False,
                                 has_data=True,
                                 is_bayesian=True,
                                 prior=prior_type(2),
                                 x_lb=x_lb,
                                 x_ub=x_ub
                                 )

resv = rmfa.process_opt_result_vector()
A = rmfa.create_mfa_matrix()

rmfa.write_result_csv(resv, "map_res")
resid = rmfa.balance_residuals(resv)
