import numpy as np
import torch
import torch.nn as nn
import yaml

from model.dcrnn_cell import DCGRUCell
import logging


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        """
        Encoder、Decoder、DCRNN 共有的属性
        :param adj_mx: np.array
        :param model_kwargs: dict
        """
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))  # DCGRUCell 的参数
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))  # 用于 Scheduled Sampling 中概率 p 的递减

        # 选择过滤器的类型，我的代码中只实现了双向
        # self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))  # 可以不需要，可以在 forward 中通过 inputs 来获取

        # GRU的层数，若Encoder有两层GRU，则说明Encoder中有两个GRU单元，然后所有输入共享两个单元的参数，即在两个单元中循环输入输出
        # Encoder 和 Decoder 的参数
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))

        # 隐藏状态的特征维度，DCGRUCell 的参数
        self.num_units = int(model_kwargs.get('num_units'))
        # self.hidden_state_size = self.num_nodes * self.num_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # 因为有两个父类，所以需要分开进行初始化
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)

        self.input_dim = int(model_kwargs.get('input_dim', 1))  # DCGRUCell 的参数
        self.seq_len = int(model_kwargs.get('seq_len'))  # 历史序列的长度
        # GRU的层数
        # 注意，不能用Python列表存储，ModuleList中的模块会被自动注册到模型中，Pyhton列表中的模块则不会
        # 注意！第一层的输入维度为 input_dim，第二层的输入维度为 num_units，输出维度都是 num_units
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.input_dim if i == 0 else self.num_units, self.num_units, adj_mx, self.max_diffusion_step)
             for i in range(self.num_rnn_layers)]
        )

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.
        这是某一个时间的输入所经过的计算
        :param inputs: shape (batch_size, num_nodes, input_dim)
                    某一个时间的输入数据
        :param hidden_state: (num_layers, batch_size, num_nodes, num_units)
                    上一个时间输出的隐藏状态
                    optional, zeros if not provided
        :return:
            output:       (batch_size, num_nodes, num_units)
                在 Encoder 中 output 用不到
            hidden_state: (num_layers, batch_size, num_nodes, num_units)
                作为下一个时间的 hidden_state，用的还是同一个 dcgru，共享参数
                (lower indices mean lower layers) 小的下标对应小的层
        """
        batch_size, num_nodes, input_dim = inputs.shape

        # Encoder 历史序列的第一个时间下输入的隐藏状态为 0
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, num_nodes, self.num_units),
                                       device=inputs.device)

        # 这一时间的隐藏状态（有 num_layers 个）
        hidden_states = []
        # Encoder 中 output 用不到，只需要 Decoder 中的 outputs
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            # output                  (batch_size, num_nodes, input_dim)
            # hidden_state[layer_num] (batch_size, num_nodes, num_units)
            # next_hidden_state       (batch_size, num_nodes, num_units)
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # 因为有两个父类，所以需要分开进行初始化
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)

        self.output_dim = int(model_kwargs.get('output_dim', 1))  # DCGRUCell 的参数
        self.horizon = int(model_kwargs.get('horizon', 1))  # 预测序列的长度
        self.prediction_layer = nn.Linear(self.num_units, self.output_dim)  # 输出要再经过一层全连接
        # 注意！第一层的输入维度为 output_dim，第二层的输入维度为 num_units，输出维度都是 num_units
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.output_dim if i == 0 else self.num_units, self.num_units, adj_mx, self.max_diffusion_step)
             for i in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.
        这是某个未来时间的预测过程
        :param inputs: shape (batch_size, num_nodes, input_dim)
                    上一个时间的输出 output（或者是真实值 ground truth，根据概率 p 来选择，Schedule Sampling）
        :param hidden_state: (num_layers, batch_size, num_nodes, num_units) optional, zeros if not provided
                    上一个时间输出的隐藏状态
        :return:
            output:       (batch_size, num_nodes, output_dim)
                Decoder 需要用到 output，还需要经过一层全连接
                根据概率 p 被选为下一个时间的输入 inputs
            hidden_state: (num_layers, batch_size, num_nodes, num_units)
                作为下一个时间的 hidden_state，用的还是同一个 dcgru，共享参数
                (lower indices mean lower layers) 小的下标对应小的层
        """
        batch_size, num_nodes, input_dim = inputs.shape

        # Decoder 预测序列的第一个时间下输入的隐藏状态为 Encoder 最后一个时间下输出的隐藏状态
        # 这一时间的隐藏状态（有 num_layers 个）
        hidden_states = []
        # 预测序列中这个时间下预测的值，其会以 p 的概率被选为下一时间下的输入
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            # output                  (batch_size, num_nodes, input_dim)
            # hidden_state[layer_num] (batch_size, num_nodes, num_units)
            # next_hidden_state       (batch_size, num_nodes, num_units)
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # output     (batch_size, num_nodes, num_units)
        # after view (batch_size * num_nodes, num_units)
        # predict    (batch_size * num_nodes, output_dim)
        predict = self.prediction_layer(output.view(-1, self.num_units))  # 输出经过一层全连接，得到预测结果
        # output (batch_size, num_nodes, output_dim)
        output = predict.view(batch_size, num_nodes, self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)

        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        # 用于 Scheduled Sampling 中概率 p 的递减
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        # 即是否使用 Schedule Sampling
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def count_parameters(self):
        """
        计算可训练的参数个数，tensor 中的元素个数
        :param model: 要计算参数的模型
        :return:
            int，参数个数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _compute_sampling_threshold(self, batches_seen):
        """
        根据已经训练过的 batch 数，来计算采样的概率 p（以 p 的概率选择真实值）
        p 随着 batches_seen 的增加逐渐减小，即逐渐选择预测的值作为下一时间的输入
        :param batches_seen: int，已经训练过的 batch 数
        :return:
            阈值 p 的范围: (0, 1)
            其中 batches_seen: 0 -> sum_batches 增大
                            p: 1 -> ...(<1)     减小
        """
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        Encoder forward pass on t time steps
        Encoder 进行 seq_len 次的前向计算，历史序列长度
        :param inputs: (batch_size, seq_len, num_nodes, input_dim)
        :return:
            encoder_hidden_state: (num_layers, batch_size, num_nodes, num_units)
        """
        # Encoder 历史序列的第一个时间下输入的隐藏状态为 0
        encoder_hidden_state = None

        # 对所有时间进行前向计算
        for t in range(self.encoder_model.seq_len):
            # 输入
            # inputs[:, t, :, :]   (batch_size, num_nodes, input_dim)
            # encoder_hidden_state (num_layers, batch_size, num_nodes, num_units)
            # 输出
            # encoder_hidden_state (num_layers, batch_size, num_nodes, num_units)
            _, encoder_hidden_state = self.encoder_model(inputs[:, t, :, :], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        Decoder 进行 horizon 次的前向计算，预测序列长度
        :param encoder_hidden_state: (num_layers, batch_size, num_nodes, num_units)
                    Decoder 预测序列的第一个时间下输入的隐藏状态为 Encoder 最后一个时间下输出的隐藏状态
        :param labels: (batch_size, horizon, num_nodes, output_dim)
                    可选，不存在则用预测作为下一时间的输入
        :param batches_seen: int, global step 已经训练过的 batch 数
                    可选，不存在则用预测作为下一时间的输入，其为 None，则说明直接用预测作为下一时间的输入
                    只有在 train 阶段才可能会用 batches_seen
        :return:
            output: (batch_size, horizon, num_nodes, output_dim)
                预测序列
        """
        num_layers, batch_size, num_nodes, num_units = encoder_hidden_state.shape

        # Decoder 第一个时间下的输入为 0
        go_symbol = torch.zeros((batch_size, num_nodes, self.decoder_model.output_dim),
                                device=encoder_hidden_state.device)
        decoder_input = go_symbol
        # Decoder 预测序列的第一个时间下输入的隐藏状态为 Encoder 最后一个时间下输出的隐藏状态
        decoder_hidden_state = encoder_hidden_state

        # 预测序列
        outputs = []

        # 对所有时间进行前向计算
        for t in range(self.decoder_model.horizon):
            # 输入
            # decoder_input        (batch_size, num_nodes, output_dim)
            # decoder_hidden_state (num_layers, batch_size, num_nodes, num_units)
            # 输出
            # decoder_output       (batch_size, num_nodes, output_dim)
            # decoder_hidden_state (num_layers, batch_size, num_nodes, num_units)
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)

            # 预测值可能作为下一时间的输入
            decoder_input = decoder_output

            # 将当前预测结果添加到预测序列中
            outputs.append(decoder_output)

            # TODO self.training在哪里？
            # if self.training and self.use_curriculum_learning:
            # 进行 Schedule Sampling，选择下一时间的输入（真实值还是预测值）
            if batches_seen is not None and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    # 若小于小于阈值 p，则选择真实值 ground truth 作为下一时间的输入，否则选择预测值
                    # p 随着 batched_seen 的增大逐渐减小
                    # 即选择预测值的概率逐渐增大
                    decoder_input = labels[:, t, :, :]

        # outputs (batch_size, horizon, num_nodes, output_dim) 在1维度进行堆叠
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: (batch_size, seq_len, num_nodes, input_dim)
        :param labels: (batch_size, horizon, num_nodes, output_dim)
                    可选，不存在则用预测作为下一时间的输入
        :param batches_seen: batches seen till now 已经训练过的 batch 数
                    可选，不存在则用预测作为下一时间的输入，其为 None，则说明直接用预测作为下一时间的输入
                    只有在 train 阶段才可能会用 batches_seen
        :return:
            output: (batch_size, horizon, num_nodes, output_dim)
        """
        # encoder_hidden_state (num_layers, batch_size, num_nodes, num_units)
        # Encoder 最后一个时间输出的隐藏状态
        encoder_hidden_state = self.encoder(inputs)
        self._logger.debug("Encoder complete, starting Decoder")

        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")

        # # 计算可训练的参数个数，tensor 中的元素个数
        # if batches_seen == 0:
        #     self._logger.info(
        #         "Total trainable parameters {}".format(count_parameters(self))
        #     )

        return outputs


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(r"D:\1_program\0_Traffic_Predition\DCRNN_My\configs\dcrnn_pems04.yaml", 'r', encoding="utf-8") as f:
        kwargs = yaml.safe_load(f)
        data_kwargs = kwargs.get('data')
        model_kwargs = kwargs.get('model')
        train_kwargs = kwargs.get('train')

        num_nodes = model_kwargs.get('num_nodes')
        adj_mx = np.random.randn(num_nodes, num_nodes).astype(np.float32)
        logger = logging.getLogger(__name__)

        dcrnn_model = DCRNNModel(adj_mx, logger, **model_kwargs).to(device)

        batch_size = data_kwargs.get('batch_size')
        input_dim = model_kwargs.get('input_dim')
        output_dim = model_kwargs.get('output_dim')
        seq_len = model_kwargs.get('seq_len')
        horizon = model_kwargs.get('horizon')
        batch_seen = 0
        x = torch.randn([batch_size, seq_len, num_nodes, input_dim]).to(device)
        y = torch.randn([batch_size, horizon, num_nodes, output_dim]).to(device)

        output = dcrnn_model(x, y, batch_seen)  # batch_size: 22 -> 32, used_gpu: 1820MB -> 2344MB, my_gpu: 4096MB
        pass
