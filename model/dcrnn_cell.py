import numpy as np
import torch
import torch.nn as nn
import utils


class DConv(nn.Module):
    # 类属性
    # 让所有的实例共享supports（保存双向矩阵D^(-1)A），减少所用内存空间
    # 并且只有在创建第一个实例时，才会初始化supports（用modified实现）
    supports = []
    modified = False

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
        # 保证所有实例只计算一次，所有实例共享一个supports
        if not __class__.modified:
            # TODO：这里没有转置，源码中还进行了一次转置
            # 经过实验，外面再加一个转置，效果更好
            supports = [utils.calculate_random_walk_matrix(adj_mx).T, utils.calculate_random_walk_matrix(adj_mx.T).T]
            for support in supports:
                # 将矩阵转换为sparse_coo_tensor格式，这样torch才可以处理
                # 注意，这里我没有进行device的设置
                __class__.supports.append(self._build_sparse_matrix(support))
            __class__.modified = True

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
        :return:
            Output 卷积后的结果: (batch_size, num_nodes, output_dim)
        """
        # 转换到相应的device中
        for i in range(len(__class__.supports)):
            __class__.supports[i] = __class__.supports[i].to(inputs.device)

        # inputs (batch_size, num_nodes, input_dim)
        batch_size, num_nodes, input_dim = inputs.shape

        # 先计算所有的多项式（D^(-1)AX）
        # 双向，每一向有 max_diffusion_step，再加上 X 自身，所以共 num_matrices = max_diffusion_step * 2 + 1 个
        # (num_nodes, input_dim, batch_size)
        x0 = inputs.permute(1, 2, 0)
        # (num_nodes, input_dim * batch_size)
        x0 = torch.reshape(x0, shape=[num_nodes, input_dim * batch_size])

        # (1, num_nodes, input_dim * batch_size)，X 自身 1 个
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            # 双向，带 D^(-1)A 的共 max_diffusion_step * 2 个
            for support in __class__.supports:
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
        num_matrices = len(__class__.supports) * self._max_diffusion_step + 1  # Adds for x itself.

        # 然后再左乘权重矩阵 weights (input_dim * num_matrices, output_dim)
        # 先将 x 的维度转换为与 weights 维度相适应的矩阵 w (input_dim * num_matrices, output_dim)
        # x (num_matrices, num_nodes, input_dim, batch_size)
        x = torch.reshape(x, shape=[num_matrices, num_nodes, input_dim, batch_size])
        # x (batch_size, num_nodes, input_dim, num_matrices)
        x = x.permute(3, 1, 2, 0)
        # x (batch_size * num_nodes, input_dim * num_matrices)
        x = torch.reshape(x, shape=[batch_size * num_nodes, input_dim * num_matrices])

        # x (batch_size * num_nodes, input_dim * num_matrices)
        # w (input_dim * num_matrices, output_dim)
        # xw -> x (batch_size * num_nodes, output_dim)
        try:
            x = torch.matmul(x, self.weights)
        except:
            pass

        # b (1, output_dim)
        if self._bias_start is not None:
            x += self.biases

        # x (batch_size * num_nodes, output_dim) -> (batch_size, num_nodes, output_dim)
        x = torch.reshape(x, shape=[batch_size, num_nodes, self._output_dim])

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


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, non_linearity='tanh'):
        """
        :param input_dim: int，输入维度
        :param num_units: int，隐藏状态的维度
        :param adj_mx: np.array
        :param max_diffusion_step: int，最大扩散步数
        :param non_linearity: str，非线性激活，计算即时状态 c 时用到
        """
        super(DCGRUCell, self).__init__()

        self._input_dim = input_dim
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._activation = torch.tanh if non_linearity == 'tanh' else torch.relu

        # 注意，bias 放在了 DConv 中
        # dconv1 用于计算重置门和更新门 [r, u] 两个，所以其隐藏单元数要 * 2
        self.d_conv1 = DConv(input_dim + num_units, num_units * 2, adj_mx, max_diffusion_step, 1.0)
        # dconv2 用于计算即时状态 c 一个
        self.d_conv2 = DConv(input_dim + num_units, num_units, adj_mx, max_diffusion_step, 0.0)

    def forward(self, inputs, hidden_state):
        """
        融合卷积的GRU
        :param inputs:       x (batch_size, num_nodes, input_dim)
        :param hidden_state: h (batch_size, num_nodes, num_units)
        :return:
            new_hidden_state: (batch_size, num_nodes, num_units)
        """
        # 将输入和上一层的隐藏状态连接，横向
        # [x, h] (batch_size, num_nodes, input_dim + num_units)
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        # 同时计算重置门 r 和更新门 u，每个门的维度和隐藏状态 h 的维度一样
        # [r, u] = sigmoid(d_conv1([x, h]))
        # [r, u] (batch_size, num_nodes, (2 * num_units))
        concatenation = torch.sigmoid(self.d_conv1(concatenation))

        # 拆分开来，每一份 num_units 个
        # r (batch_size, num_nodes, num_units)
        # u (batch_size, num_nodes, num_units)
        r, u = torch.split(concatenation, self._num_units, dim=2)

        # [x, (r * h)] (batch_size, num_nodes, input_dim + num_units)
        concatenation = torch.cat((inputs, r * hidden_state), dim=2)

        # 计算即时状态 c
        # c = tanh(d_conv2([x, (r * h)]))，激活函数默认为 tanh
        # c (batch_size, num_nodes, num_units)
        c = self._activation(self.d_conv2(concatenation))

        # 计算该层的隐藏状态
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes, num_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c

        return new_hidden_state


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
