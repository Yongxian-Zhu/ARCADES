# -*- coding: utf-8 -*-

import pytensor
pytensor.config.cxx = "/usr/bin/clang++"

import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import nltk
# nltk.download('stopwords')

import pandas as pd
import numpy as np

import Levenshtein
import sys

from pyomo.environ import *
import pymc as pm

def find_unique_strings(string_list, threshold=3):
    """
    Finds unique strings in a list by grouping similar ones.

    Args:
        string_list: A list of strings.
        threshold: The maximum Levenshtein distance for strings to be considered the same.

    Returns:
        A list of representative unique strings.
    """
    if not string_list:
        return []

    unique_groups = []
    group_id = []
    for current_string in string_list:
        found_group = False
        for i, group in enumerate(unique_groups):
            # Compare the current string to the representative of each group
            if Levenshtein.distance(current_string.lower(), group[0].lower()) <= threshold:
                unique_groups[i].append(current_string)
                group_id.append(i)
                found_group = True
                break

        if not found_group:
            unique_groups.append([current_string])
            group_id.append(len(unique_groups)-1)

    # Return the representative string from each group
    #return [group[0] for group in unique_groups]
    print("Levenshtein distance:")
    print(f"Length of original keys {len(string_list)}")
    print(f"Length of unique group {len(unique_groups)}")
    return [unique_groups[i][0] for i in group_id]


#my_list = ["Hall-Héroult", "Hall-Héroult", "Hall-Hauroult", "Hal-Héroult", "Aluminum", "Aluminium"]
#unique_entries = find_unique_strings(my_list)
#print(unique_entries)


def tokenize_and_normalize(keys):
    """
    Tokenize and normalize a list of keys.

    Args:
        keys (list): List of string keys to process.

    Returns:
        list: List of normalized tokens for each key.
    """
    #keys = find_unique_strings(keys)


    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    # remove 'other'
    stop_words = set(stopwords.words('english')).difference({"other"})
    #print(stop_words)

    normalized_keys = []

    for key in keys:
        # Step 1: Tokenize using regular expressions (split by non-alphanumeric characters)
        tokens = re.split(r'\W+', key)

        # Step 2: Normalize tokens (lowercase, remove stopwords, stem)
        normalized_tokens = []
        for token in tokens:
            new_token = stemmer.stem(token.lower())
            # we want at least one item
            if len(tokens) == 1 or token and token.lower() not in stop_words:
                normalized_tokens.append(new_token)
            #else:
            #    normalized_tokens.append(token)
        #
        #normalized_tokens = [
        #    stemmer.stem(token.lower())
        #    for token in tokens
        #    if token and token.lower() not in stop_words
        #]

        normalized_keys.append(normalized_tokens)

    # create a new list of keys separated with a "_"
    new_keys = []
    for k in normalized_keys:
        k_new = "_".join(k)
        k_new.strip("_")
        new_keys.append(k_new)


    return new_keys

class ReducedSpaceMfa:
    def __init__(self, mfa_xlsx: str):
        self.mfa_xlsx = mfa_xlsx
        self.n_node = 1
        self.n_arc = 1
        self.inc_matrix = np.zeros((1,1))
        self.node_type = pd.DataFrame([1], columns=["a"])
        self.variable_names = pd.DataFrame([1], columns=["a"])
        self.n_dataset = 0
        self.data_matrix = np.zeros((1, 1))
        self.data_flag = np.full((1, 1), False, dtype=bool)
        self.score_matrix = np.zeros((1, 1))

    # ##########################################################################

    def pre_process_graph(self, input_file):

        dict_types = {'from_node_name': str,
                      'to_node_name': str,
                      'from_node_number': np.int64,
                      'to_node_number': np.int64,
                      'Value 1': np.float64,
                      'Value 2': np.float64,
                      'Value 3': np.float64
                      }
        df = pd.read_csv(input_file, dtype=dict_types)

        keys_0 = df["from_node_name"].to_list()
        new_keys0 = tokenize_and_normalize(keys_0)
        keys_1 = df["to_node_name"].to_list()
        new_keys1 = tokenize_and_normalize(keys_1)

        arc_list = [k for k in zip(new_keys0, new_keys1)]
        unique_arcs = set(arc_list)

        inconsistent_arcs = len(arc_list) - len(unique_arcs)

        df["from_node_name"] = new_keys0
        df["to_node_name"] = new_keys1
        df.to_csv("example.csv")

        if inconsistent_arcs > 0:
            seen = set()
            duplicates = set()
            for i, name in enumerate(arc_list):
                if name in seen:
                    duplicates.add(name)
                else:
                    seen.add(name)

            print("Duplicates")
            print(duplicates)

            raise Exception("The flow(arc) list provided is inconsistent\n"
                            f"by {inconsistent_arcs}")

        else:
            n_arc = len(unique_arcs)



        # start nodes have "from" but not "to"
        start_nodes = set(new_keys0).difference(new_keys1)
        end_nodes = set(new_keys1).difference(new_keys0)

        ds_nodes = start_nodes.union(end_nodes)



        node_list = list(set(new_keys0).union(new_keys1))

        n_node = len(node_list)
        # inc_matrix has the row order of the node_list and the column order of
        # the arc list, i.e. the dataframe
        inc_matrix = np.zeros((n_node, n_arc), dtype=np.int8)

        # arc id is given by the position in the list
        for arc_id, row in df.iterrows():
            node_i = node_list.index(row.loc["from_node_name"])
            node_j = node_list.index(row.loc["to_node_name"])
            inc_matrix[node_i, arc_id] = -1 # from is -1
            inc_matrix[node_j, arc_id] = 1

        self.n_node = n_node
        self.n_arc = n_arc
        self.inc_matrix = inc_matrix
        self.ds_nodes = ds_nodes
        self.node_list = node_list
        self.df = df


    # ##########################################################################
    def write_problem_csvs(self):
        self.node_type.to_csv("node_type.csv")
        dinc.to_csv("inc_matrix.csv")
        dnames.to_csv("variable_names.csv")

    # ##########################################################################
    def read_data(self,
                  n_dataset,
                  start_column,
                  has_scores: bool=False,
                  score_file: str=""):
        # infer the number of datasets
        # assume: from_node_name to_node_name from_node_number to_node_number...
        df = self.df
        data_matrix = np.zeros((self.n_arc, n_dataset))
        data_flag = np.full((self.n_arc, n_dataset), False, dtype=bool)

        # we need to find the position in the dataset
        for index, rv in df.iterrows():
            # in this case the flow has the same order
            # look at each column
            for ds in range(n_dataset):
                col = df.columns.get_loc("Value 1") + ds
                if not(pd.isna(rv.iloc[col])):
                    data_matrix[index, ds] = rv.iloc[col]
                    data_flag[index, ds] = True

        self.n_dataset = n_dataset
        self.data_matrix = data_matrix
        self.data_flag = data_flag
        #
        if has_scores:
            #score_file =
            df_scores = pd.read_csv(score_file)
            if df.shape[0] != df_scores.shape[0]:
                raise Exception("The scores and the data are inconsistent.")

            self.score_matrix = np.zeros((self.n_arc, self.n_dataset))

            i = 0
            for index, rv in df_scores.iterrows():
                for ds in range(n_dataset):
                    if not(pd.isna(rv.iloc[ds])):
                        self.score_matrix[index, ds] = rv.iloc[ds]

    # ##########################################################################
    def read_problem_graph(self) -> tuple:
        self.inc_matrix = pd.read_csv("inc_matrix.csv",
                                       index_col=0,
                                       header=0,
                                       dtype=np.int8)
        self.inc_matrix = self.inc_matrix.to_numpy()

        node_type = pd.read_csv("node_type.csv",
                                index_col=0,
                                dtype={"type": np.int8, "number": np.int8,
                                       "name": str})

        variable_names = pd.read_csv("variable_names.csv", index_col=0)

        return incidence_matrix, node_type, variable_names

    # ##########################################################################
    def run_parametric_optimization(self, is_q_param: bool,
                                    q_val: np.ndarray=np.zeros(1),
                                    has_data: bool=False):

        inc_matrix = self.inc_matrix
        node_type = self.node_type
        variable_names = self.variable_names

        m = ConcreteModel()

        n_node, n_arc = inc_matrix.shape

        m.arcs = Set(initialize=range(n_arc))
        m.nodes = Set(initialize=range(n_node))

        m.flow = Var(m.arcs, bounds=(0, None))
        # put this residual as temporarily
        m.residual = Var(m.nodes, initialize=0)

        # we want a sparse index only for the q in the ds_node list
        m.q_set = Set(initialize=[i for i in range(n_node) if self.node_list[i]
                                  in self.ds_nodes]
                      )

        if is_q_param:
            m.q = Param(m.q_set, initialize=q_val)
        else:
            m.q = Var(m.q_set, initialize=0.0)


        def flow_con(m, no):
            if sum(np.abs(inc_matrix[no, :])) == 0:
                print(f"Node {no} balance needs checking?")
                return Constraint.Skip
            else:
                if self.node_list[no] in self.ds_nodes:
                    return sum(inc_matrix[no, ar] * m.flow[ar] for ar in m.arcs if
                               inc_matrix[no,ar] != 0) == m.q[no]
                else:
                    return sum(inc_matrix[no, ar] * m.flow[ar] for ar in m.arcs if
                               inc_matrix[no,ar] != 0) == 0.0

        m.flow_con = Constraint(m.nodes, rule=flow_con)

        if has_data:
            # expression for convenience
            def likelihood_expr_(m, i, k):
                #return m.flow[i] - f_data0[i]
                f1 = np.vectorize(lambda x: 0.4 * (1.5 - x))
                sigma_0 = np.multiply(f1(self.score_matrix), self.data_matrix)
                min_v_1e_3 = np.vectorize(lambda x: max(x, 1e-03))
                sigma_0 = min_v_1e_3(sigma_0)

                if self.data_flag[i, k]:
                    return (m.flow[i] - self.data_matrix[i, k])**2/sigma_0[i, k]
                else:
                    return Expression.Skip

            # self.n_dataset = n_dataset
            m.data_set = Set(initialize=range(self.n_dataset))
            m.likelihood_expr = Expression(m.arcs, m.data_set,
                                           expr=likelihood_expr_)

            m.obj_fun = Objective(rule=(sum(m.likelihood_expr[i, k]
                                            for i in m.arcs for k in
                                            m.data_set if self.data_flag[i, k])
                                        ))
            # Objective
        else:
            # least squares

            def res_expr(m, no):
                if sum(np.abs(inc_matrix[no, :])) == 0:
                    return Expression.Skip
                else:
                    if self.node_list[no] in self.ds_nodes:
                        return sum(inc_matrix[no, ar] * m.flow[ar] for ar in m.arcs if
                                   inc_matrix[no,ar] != 0) - m.q[no]
                    else:
                        return sum(inc_matrix[no, ar] * m.flow[ar] for ar in m.arcs if
                                   inc_matrix[no,ar] != 0)

            m.res_expr = Expression(m.nodes, expr=res_expr)

            m.obj_fun = Objective(rule=(sum(m.res_expr[no]**2 for no in m.nodes)))



        ipexe = "/Users/dthierry/Apps/ipopt_dir/bin/ipopt"

        solver = SolverFactory(ipexe)
        # solver.options["outlev"] = 4


        # Solve
        solver.solve(m, tee=True)
        self.model = m

        non_stale_count = 0
        for ar in m.arcs:
            if not(m.flow[ar].stale):
                non_stale_count += 1
        for no in m.nodes:
            if self.node_list[no] in self.ds_nodes:
                if not(m.q[no].stale):
                    non_stale_count += 1

        supposed_var_len = self.n_arc + len(self.ds_nodes)
        if non_stale_count != supposed_var_len: #len(self.dname):
            print("The size of the variables is different from the graph")
            print(f"len dname = {len(self.dname)}\t nonstale vars={non_stale_count}")

        result_vector = np.full(supposed_var_len, -9090909.0)
        for row in range(self.n_arc):
            if m.flow[row].stale:
                print(f"flow variable {row} is stale")
            else:
                v = value(m.flow[row])
                result_vector[row] = v
        q_id = 0
        for i, no in enumerate(self.node_list):
            if no in self.ds_nodes:
                if m.q[i].stale:
                    print(f"q variable {i} is stale!") # it should not
                else:
                    row = self.n_arc + q_id
                    v = value(m.q[i])
                    result_vector[row] = v
                    q_id += 1

        return result_vector

    # ##########################################################################
    def create_mfa_matrix(self):#, fixed_q_dict: dict):
        inc_matrix = self.inc_matrix
        node_type = self.node_type
        variable_names = self.variable_names

        n_node, n_arc = inc_matrix.shape

        # iterate by column
        count = 0

        n_rows = n_node

        # flows + q's
        n_cols = n_arc + len(self.ds_nodes)

        A = np.zeros((n_rows, n_cols))

        print(f"A original shape {A.shape}")

        rhs = np.zeros(n_rows)

        # matrix
        # by column (f block)
        for arc in range(n_arc):
            # by row (down)
            for row in range(n_node):
                A[row, arc] += inc_matrix[row, arc]

        # by column (q block), i.e. the last n_arc + q_idx columns
        q_idx = 0
        for node in range(n_node):
            if self.node_list[node] in self.ds_nodes:
                col = n_arc + q_idx
                row = node
                A[row, col] += -1
                q_idx += 1

        self.A_mfa = A
        return A

    def reduced_space_matrices(self):
        r"""
        Idea: Given
        .. math::
            A * x = b,
        where $A\in \mathbb{R}^m*n$ ,
        then let us create the following matrix:
        .. math::
            [Y Z] \in \mathbb{R}^n*n : nonsingular, and
            A * Z = 0
        Where $Y \in \mathbb{R}^n*m and Z \in \mathbb{R}^n*(n-m)$
        Then we express x as x = Y * x_y + Z * x_z
        Then let x = Y*(A*Y)^-1 * b + Z*x_z
        """

        # perform QR factorization
        Q, R = np.linalg.qr(self.A_mfa.transpose(), mode="complete")

        print(f"Shape of Q\t{Q.shape}")
        print(f"Shape of R\t{R.shape}")

        # q = pd.DataFrame(Q)
        # r = pd.DataFrame(R)
        # q.to_csv("q_factor.csv")
        # r.to_csv("r_factor.csv")

        m, n = self.A_mfa.shape

        # null-space
        Y = Q[:, :m]  # n by m
        Z = Q[:, m:]  # n by (n-m)

        Rm = R[:m, :]  # m by m

        #result = np.allclose(np.zeros((m, m)), np.dot(A, Z))
        # we have to sample directly on Z*x_z

        # Then let x = Y(A*Y)^-1 * b + Z*x_z
        self.Z = Z
        self.Y = Y

    def red_space_sampling(self, mu_z, sigma_z):
        n_minus_m = mu_z.size
        mu_prior = mu_z
        cov_prior = sigma_z

        data_matrix = self.data_matrix

        n_arc = self.n_arc
        n_node = self.n_node

        n_var = n_arc + len(self.ds_nodes)

        dr, dc = data_matrix.shape

        observed_matrix = np.zeros(n_var)
        # first rows are the arcs
        observed_matrix[:dr] = data_matrix[:, 1]

        sigma_noise = np.multiply(self.score_matrix, self.data_matrix)
        min_v_1e_3 = np.vectorize(lambda x: max(x, 1e-03))
        sigma_noise = min_v_1e_3(sigma_noise)
        # TODO revise this:
        vector_sqrt = np.vectorize(lambda x: 0.4 * (1.5 - x))
        sigma_noise = vector_sqrt(sigma_noise)

        print(f"Data matrix has dimensions of {dr, dc}")
        if dr != n_arc:
            raise Exception("The number of arcs is inconsistent with the data\n"
                            f"by {dr-n_arcs}")

        print(f"mu_z shape = {mu_z.shape}")
        print(f"cov_z shape = {sigma_z.shape}")
        print(f"Z shape = {self.Z.shape}")
        print(f"n_arc + n_node shape = {n_arc + n_node}")

        Z = self.Z
        # there should be a global mean
        # there should be data set mean
        # there should be data set sigma
        with pm.Model() as multivariate_model:
            # Priors for the mean vector
            mu_z = pm.MvNormal('mu_z',
                               mu=mu_prior,
                               cov=cov_prior,
                               shape=n_minus_m)

            mu = pm.Deterministic('mu', pm.math.dot(Z, mu_z)) # or use Z @ x

            sd_dist = pm.Exponential.dist(1.0, shape=n_var)
            L, _, _ = pm.LKJCholeskyCov("chol_cov",
                                        n=n_var,
                                        eta=2.0,
                                        sd_dist=sd_dist,
                                        compute_corr=True)

            #L = pm.expand_packed_triangular(n_var, packed_L)

            # first n_arc elements are the flows

            #obs = pm.MvNormal("obs", mu=mu, chol=L, observed=data)
            # TODO: have all datasets and the scores.
            likelihood = pm.MvNormal('likelihood', mu=mu, chol=L,
                                     observed=observed_matrix)

            idata = pm.sample(draws=2000, tune=2000, chains=4, cores=1, target_accept=0.9, return_inferencedata=True)
        return idata

        #print(az.summary(trace, var_names=["mu_z", "mu"]))


    def simple_data_variance(self):
        data_matrix = self.data_matrix
        data_flag = self.data_flag
        nrow, ncol = data_matrix.shape
        n_var = self.n_arc + len(self.ds_nodes)
        variance = np.ones(n_var)*1e-03
        for row in range(nrow):
            if data_flag[row, :].sum() > 1:
                s = sum(data_matrix[row, col] for col in range(ncol) if data_flag[row, col])
                mu = s / data_flag[row, :].sum()
                s = sum((data_matrix[row, col] - mu)**2 for col in range(ncol) if
                        data_flag[row, col])
                variance[row] = s / data_flag[row, :].sum()
        return variance


    def red_space_mult_data_sampling(self, mu_z, sigma_z):
        n_minus_m = mu_z.size
        mu_prior = mu_z
        #cov_prior = sigma_z
        L_cholesky_cov = np.linalg.cholesky(sigma_z)
        print("Cholesky")
        print(L_cholesky_cov.shape)
        data_matrix = self.data_matrix

        n_arc = self.n_arc
        n_node = self.n_node

        n_var = n_arc + len(self.ds_nodes)

        dr, dc = data_matrix.shape

        observed_matrix = np.zeros(n_var)
        # first rows are the arcs
        observed_matrix[:dr] = data_matrix[:, 1]


        f1 = np.vectorize(lambda x: 0.4 * (1.5 - x))
        sigma_noise = np.multiply(f1(self.score_matrix), self.data_matrix)
        min_v_1e_3 = np.vectorize(lambda x: max(x, 1e-03))
        sigma_noise = min_v_1e_3(sigma_noise)

        print(f"Data matrix has dimensions of {dr, dc}")
        if dr != n_arc:
            raise Exception("The number of arcs is inconsistent with the data\n"
                            f"by {dr-n_arcs}")

        print(f"mu_z shape = {mu_z.shape}")
        print(f"cov_z shape = {sigma_z.shape}")
        print(f"Z shape = {self.Z.shape}")
        print(f"n_arc + n_node shape = {n_arc + n_node}")


        permutation_mat = []
        cholesky_data = []
        observation_data = []

        for ds in range(self.n_dataset):
            a_k = np.zeros((self.data_flag[:, ds].sum(), self.data_flag.shape[0]))
            new_row = 0
            for row in range(self.data_flag.shape[0]):
                if self.data_flag[row, ds]:
                    a_k[new_row, row] = 1
                    new_row += 1


            #sigma_noise_k = np.diag(sigma_noise[:, ds]) @ a_k.T
            #sigma_noise_k = a_k @ sigma_noise_k

            sigma_noise_k = np.linalg.multi_dot([a_k, np.diag(sigma_noise[:, ds]), a_k.T])


            cholesky_L_k = np.sqrt(sigma_noise_k)
            cholesky_data.append(cholesky_L_k)

            observation_k = a_k @ self.data_matrix[:, ds]
            observation_data.append(observation_k)

            a_perm = np.block([a_k, np.zeros((a_k.shape[0], len(self.ds_nodes)))])
            permutation_mat.append(a_perm)




        Z = self.Z
        # there should be a global mean
        # there should be data set mean
        # there should be data set sigma
        with pm.Model() as multivariate_model:
            # Priors for the mean vector
            # sd_dist = pm.HalfNormal.dist(1.0, shape=n_minus_m)
            # L, _, _ = pm.LKJCholeskyCov("chol_cov",
            #                             n=n_minus_m,
            #                             eta=2.0,
            #                             sd_dist=sd_dist,
            #                             compute_corr=True)

            mu_z = pm.MvNormal('mu_z',
                               mu=mu_prior,
                               chol=L_cholesky_cov,
                               # chol=L,
                               shape=n_minus_m)

            mu = pm.Deterministic('mu', pm.math.dot(Z, mu_z)) # or use Z @ x


            #obs = pm.MvNormal("obs", mu=mu, chol=L, observed=data)
            # TODO: have all datasets and the scores.
            for ds in range(self.n_dataset):
                a_perm = permutation_mat[ds]
                cholesky_L_k = cholesky_data[ds]
                observation_k = observation_data[ds]

                mu_k = pm.Deterministic(f"mu_{ds}", pm.math.dot(a_perm, mu))

                pm.MvNormal(f'L_{ds}',
                            mu=mu_k,
                            chol=cholesky_L_k,
                            observed=observation_k
                            )

            idata = pm.sample(draws=2000, tune=2000, chains=4, cores=1, target_accept=0.9, return_inferencedata=True)
        return idata

