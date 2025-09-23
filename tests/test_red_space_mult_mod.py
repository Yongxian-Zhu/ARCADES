import numpy as np
import arviz as az
import sys

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa

f = "/Users/dthierry/Projects/tue-27-may-25/data/2017_MFA_file.xlsx"
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

_, variance = rmfa.simple_data_moments()
print(f"variance shape = {variance.shape}")
print(f"Z shape = {rmfa.Z.shape}")
print(f"Zpseudo shape = {Zpseudo.shape}")

variance_z = np.diag(variance) @ Zpseudo.transpose()
variance_z = Zpseudo @ variance_z

print(variance_z)

z_samples = rmfa.red_space_mult_data_sampling(mu_z, variance_z)
print(az.summary(z_samples, var_names=["mu_z", "mu"]))
mean_posterior = az.summary(z_samples, var_names=["mu"])["mean"].to_frame()
mean_posterior["mu_prior"] = mu
mean_posterior.to_csv("test_result_red.csv")
