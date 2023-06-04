import scipy.sparse as sp
import numpy as np


def calculate_random_walk_matrix(adj_mx):
    """
    计算随机游走矩阵，结果返回的是COO格式的稀疏张量，因为torch支持COO
    :param adj_mx: np.array
    :return:
        random_walk_mx: coo_matrix
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


if __name__ == '__main__':
    N = 10
    adj_mx = np.random.randn(N, N)
    calculate_random_walk_matrix(adj_mx)
