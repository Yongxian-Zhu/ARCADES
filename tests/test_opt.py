import pandas as pd
import numpy as np
import sys

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa

f = "/Users/dthierry/Projects/tue-27-may-25/data/2017_MFA_file.xlsx"
f_score = "/Users/dthierry/Projects/tue-27-may-25/input_file_score.csv"
f_input = "/Users/dthierry/Projects/tue-27-may-25/input_file.csv"

rmfa = ReducedSpaceMfa(f)
rmfa.pre_process_graph(f_input)
rmfa.read_data(3, 1, has_scores=True, score_file=f_score)
A = rmfa.create_mfa_matrix()

rmfa.run_parametric_optimization(is_q_param=False,
                                        has_data=True,
                                        is_bayesian=True
                                       )

resv = rmfa.process_opt_result_vector()
inf_res = rmfa.balance_residuals(resv)
rmfa.conf_int_init()
inf_res = rmfa.rho_change(3e3)

# # compare with the reduced space sampling
# other_result_file = "test_result.csv"
# d = pd.read_csv(other_result_file)
#
# mu_prior_rs_sampling = d.iloc[:, 2].to_numpy()
# mu_posterior_rs_sampling = d.iloc[:, 1].to_numpy()
#
# # residuals
# res_opt = A @ resv
# res_RSs = A @ mu_posterior_rs_sampling
# res_RSprior = A @ mu_prior_rs_sampling
#
#
# inf_res_opt = np.linalg.norm(res_opt, np.inf)
# inf_res_RSs = np.linalg.norm(res_RSs, np.inf)
# inf_res_RSprior = np.linalg.norm(res_RSprior, np.inf)
