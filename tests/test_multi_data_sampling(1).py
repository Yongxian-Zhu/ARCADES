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

# quadratic form affine transformation of the covariance

variance = rmfa.simple_data_variance()
print(f"variance shape = {variance.shape}")
print(f"Z shape = {rmfa.Z.shape}")
print(f"Zpseudo shape = {Zpseudo.shape}")

covariance_z = np.linalg.multi_dot([Zpseudo, np.diag(variance), Zpseudo.T])

n_var = rmfa.n_arc + len(rmfa.ds_nodes)

# data has at most n_arc rows
# variables have n_var rows


f1 = np.vectorize(lambda x: 0.4 * (1.5 - x))
sigma_noise = np.multiply(f1(rmfa.score_matrix), rmfa.data_matrix)
min_v_1e_3 = np.vectorize(lambda x: max(x, 1e-03))
sigma_noise = min_v_1e_3(sigma_noise)

permutation_mat = []
cholesky_data = []
observation_data = []

for ds in range(rmfa.n_dataset):
    a_k = np.zeros((rmfa.data_flag[:, ds].sum(), rmfa.data_flag.shape[0]))
    new_row = 0
    for row in range(rmfa.data_flag.shape[0]):
        if rmfa.data_flag[row, ds]:
            a_k[new_row, row] = 1
            new_row += 1

    #sigma_noise_k = a_k @ sigma_noise[:, ds]
    #sigma_noise_k = sigma_noise_k @ a_k.T

    print(f"sigma_noise size {sigma_noise[:, ds].shape}")
    print(f"a_k size {a_k.shape}")

    sigma_noise_k = np.linalg.multi_dot([a_k, np.diag(sigma_noise[:, ds]), a_k.T])

    print(f"sigma_noise_k size {sigma_noise_k.shape}")

    cholesky_L_k = np.diag(np.sqrt(sigma_noise_k))
    cholesky_data.append(cholesky_L_k)

    observation_k = a_k @ rmfa.data_matrix[:, ds]
    observation_data.append(observation_k)

    a_perm = np.block([a_k, np.zeros((a_k.shape[0], len(rmfa.ds_nodes)))])
    permutation_mat.append(a_perm)
