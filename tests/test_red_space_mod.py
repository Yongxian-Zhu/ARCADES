import numpy as np
import arviz as az
import sys

sys.path.insert(1, "../Dynamic-Material-Flow-Analysis")

from red_space_model import ReducedSpaceMfa

f = "/Users/dthierry/Projects/tue-27-may-25/data/2017_MFA_file.xlsx"
f_score = "/Users/dthierry/Projects/tue-27-may-25/input_file_score.csv"
f_input = "/Users/dthierry/Projects/tue-27-may-25/input_file.csv"

rmfa = ReducedSpaceMfa(f)
rmfa.pre_process_graph(f_input)
rmfa.read_data(3, 1)

resv = rmfa.run_parametric_optimization(is_q_param=False, has_data=True)

A = rmfa.create_mfa_matrix()

rmfa.reduced_space_matrices()

Zpseudo = np.linalg.pinv(rmfa.Z)

# let mu = resv then we can have that the mu_z = Zpseudo @ mu
# this is true because rhs = 0
mu = resv

mu_z = Zpseudo @ mu

# quadratic form affine transformation of sigma

variance = rmfa.simple_data_variance()
print(f"variance shape = {variance.shape}")
print(f"Z shape = {rmfa.Z.shape}")
print(f"Zpseudo shape = {Zpseudo.shape}")

#variance_z = np.diag(variance) @ Zpseudo.transpose()
#variance_z = Zpseudo @ variance_z

covariance_z = np.linalg.multi_dot([Zpseudo, np.diag(variance), Zpseudo.T])

z_samples = rmfa.red_space_sampling(mu_z, covariance_z)

#
print(az.summary(z_samples, var_names=["mu_z", "mu"]))

mean_posterior = az.summary(z_samples, var_names=["mu"])["mean"].to_frame()
mean_posterior["mu_prior"] = mu
mean_posterior.to_csv("test_result.csv")
