import pandas as pd
import numpy as np
import sys

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa

f = "/Users/dthierry/Projects/tue-27-may-25/data/2017_MFA_file.xlsx"
f_score = "/Users/dthierry/Projects/tue-27-may-25/input_file_score.csv"
f_input = "/Users/dthierry/Projects/tue-27-may-25/input_file.csv"

rmfa = ReducedSpaceMfa()
rmfa.pre_process_graph(f_input)
rmfa.read_data(3, 1, has_scores=True, score_file=f_score)
A = rmfa.create_mfa_matrix()

rmfa.run_parametric_optimization(is_q_param=False,
                                        has_data=True,
                                        is_bayesian=True
                                       )

resv = rmfa.process_opt_result_vector()
A = rmfa.create_mfa_matrix()

rmfa.write_result_csv(resv, "map_res")
resid = rmfa.balance_residuals(resv)
