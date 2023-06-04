import numpy as np
import torch
import torch.nn as nn
import utils


class DConv(nn.Module):
    def __init__(self, input_dim, output_dim, adj_mx, max_diffusion_step, bias_start=0.0):
        """
        扩散卷积
        :param input_dim: int
        :param output_dim: int
        :param adj_mx: np.array，邻接矩阵
        :param max_diffusion_step: int，最大扩散步数，若为2，则最大阶数为2（0、1、2）
        :param bias_start: float，若为None，则不添加bias
        """
        super(DConv, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._max_diffusion_step = max_diffusion_step  # 最大扩散步数，若为2，则最大阶数为2（0、1、2）
        self._num_matrices = max_diffusion_step * 2 + 1  # 累加的矩阵个数，权重矩阵大小要与之对应
        self._bias_start = bias_start

        # 扩散过程所用的矩阵 D^(-1)W，两个（双向）
        self._supports = []
        # TODO：这里没有转置，源码中还进行了一次转置
        supports = [utils.calculate_random_walk_matrix(adj_mx), utils.calculate_random_walk_matrix(adj_mx.T)]
        for support in supports:
            # 将矩阵转换为sparse_coo_tensor格式，这样torch才可以处理
            # 注意，这里我没有进行device的设置
            self._supports.append(self._build_sparse_matrix(support))

        # weights (input_dim * num_matrices, output_dim)
        self.weights = nn.Parameter(
            torch.empty(self._input_dim * self._num_matrices, self._output_dim)
        )
        # 权重初始化
        nn.init.xavier_normal_(self.weights)

        # bias (1, output_dim)
        if bias_start is not None:
            self.biases = nn.Parameter(
                torch.empty(1, self._output_dim)
            )
            nn.init.constant_(self.biases, bias_start)

    def forward(self, inputs):
        """
        :param inputs: (batch_size, num_nodes, input_dim)
        :return:    x: (batch_size, num_nodes, output_dim)
        """
        # 转换到相应的device中
        for i in range(len(self._supports)):
            self._supports[i] = self._supports[i].to(inputs.device)

        # inputs (batch_size, num_nodes, input_dim)
        batch_size = inputs.shape[0]
        num_nodes = inputs.shape[1]

        # 先计算所有的多项式（D^(-1)AX）
        # 双向，每一向有 max_diffusion_step，再加上 X 自身，所以共 num_matrices = max_diffusion_step * 2 + 1 个
        # (num_nodes, input_dim, batch_size)
        x0 = inputs.permute(1, 2, 0)
        # (num_nodes, input_dim * batch_size)
        x0 = torch.reshape(x0, shape=[num_nodes, self._input_dim * batch_size])
        # (1, num_nodes, input_dim * batch_size)，X 自身 1 个
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            # 双向，带 D^(-1)A 的共 max_diffusion_step * 2 个
            for support in self._supports:
                # support (num_nodes, num_nodes)
                # x0 (num_nodes, input_dim * batch_size)
                # x1 (num_nodes, input_dim * batch_size)
                x1 = torch.sparse.mm(support, x0)
                # x (2, num_nodes, input_dim * batch_size)
                x = torch.cat([x, x1.unsqueeze(0)], dim=0)

                for k in range(2, self._max_diffusion_step + 1):
                    # 切比雪夫多项式递推公式
                    # x2 (num_nodes, input_dim * batch_size)
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    # x (k + 1, num_nodes, input_dim * batch_size)
                    x = torch.cat([x, x2.unsqueeze(0)], dim=0)
                    x1, x0 = x2, x1

        # x (max_diffusion_step * 2 + 1, num_nodes, input_dim * batch_size), *2表示双向扩散, +1表示k=0时的x自身
        # num_matrices = max_diffusion_step * 2 + 1，多项式的个数
        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.

        # 然后再左乘权重矩阵 weights (input_dim * num_matrices, output_dim)
        # 先将 x 的维度转换为与 weights 维度相适应的矩阵 w (input_dim * num_matrices, output_dim)
        # x (num_matrices, num_nodes, input_dim, batch_size)
        x = torch.reshape(x, shape=[num_matrices, num_nodes, self._input_dim, batch_size])
        # x (batch_size, num_nodes, input_dim, num_matrices)
        x = x.permute(3, 1, 2, 0)
        # x (batch_size * num_nodes, input_dim * num_matrices)
        x = torch.reshape(x, shape=[batch_size * num_nodes, self._input_dim * num_matrices])

        # x (batch_size * num_nodes, input_dim * num_matrices)
        # w (input_dim * num_matrices, output_dim)
        # xw -> x (batch_size * num_nodes, output_dim)
        x = torch.matmul(x, self.weights)

        # b (1, output_dim)
        if self._bias_start is not None:
            x += self.biases

        # x (batch_size * num_nodes, output_dim) -> (batch_size, num_nodes, output_dim)
        x = torch.reshape(x, [batch_size, num_nodes, self._output_dim])

        return x

    @staticmethod
    def _build_sparse_matrix(L):
        """
        将coo_matrix转换为torch中的coo_tensor
        :param L: scipy.sparse中的类型，如coo_matrix
        :return:
            L: sparse_coo_tensor
        """
        L = L.tocoo()  # 确保是coo_matrix，注意只能将scipy.sparse中的稀疏转为coo，不能直接将np.array转为coo
        indices = np.column_stack((L.row, L.col))  # 对列进行合并，indices [[0, 0], [0, 1], [1, 1], [0, 2]]
        indices = torch.from_numpy(indices)
        # # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        # indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        # TODO：源码中还进行了上述的排序，但是我认为应该不用排序了，排序的话就和L.data的顺序对不上了
        # 注意，这里我没有进行device的设置
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape)
        return L


# class DCGRUCell(nn.Module):
#     def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, nonlinearity='tanh'):
#         super(DCGRUCell, self).__init__()
#         self._input_dim = input_dim
#         self._num_units = num_units
#         self._max_diffusion_step = max_diffusion_step
#         self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
#
#         self.dconv1 = DConv(input_dim + num_units, num_units * 2, adj_mx, )


if __name__ == '__main__':
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N1 = 207
    B1 = 64

    input_dim1 = 2
    output_dim1 = 64
    adj_mx1 = np.random.randn(N1, N1).astype(np.float32)
    max_diffusion_step1 = 2
    dconv = DConv(input_dim1, output_dim1, adj_mx1, max_diffusion_step1)

    dconv = dconv.to(device1)

    inputs1 = torch.randn([B1, N1, input_dim1], device=device1)

    outputs1 = dconv(inputs1)
