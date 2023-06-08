import os
import time
from tqdm import tqdm

import numpy as np
import torch

import utils
from utils import data
from utils.loss import masked_mae_loss
from utils import metrics
from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        device = self._train_kwargs.get('device', "cuda")
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

        # 用于梯度裁剪，只解决梯度爆炸问题，不解决梯度消失问题。
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # 数据集的名称，用于输出路径中的命名 runs/PEMS04/train
        self.dataset_name = self._data_kwargs.get('data_file').split('/')[1]

        # 该次运行结果所在的路径，最大为 runs/PEMS04/train100
        self.runs_path = "runs/" + self.dataset_name + "/train100/"
        for i in range(1, 100):
            self.runs_path = "runs/" + self.dataset_name + "/train{}/".format(i)
            if not os.path.exists(self.runs_path):
                os.makedirs(self.runs_path)
                break
        if not os.path.exists(self.runs_path):
            os.makedirs(self.runs_path)

        # logging.
        # 日志参数
        # self._log_dir = self._get_log_dir(kwargs)
        # self._writer = SummaryWriter('runs/' + self._log_dir)
        self._log_dir = self.runs_path
        log_level = self._kwargs.get('log_level', 'INFO')  # 获取日志级别，默认为INFO
        # 创建logger，__name__表示运行文件名
        self._logger = utils.get_logger(log_dir=self._log_dir, name=__name__,
                                        log_filename='info.log', level=log_level)

        # 每训练 save_epoch，则保存一次模型
        self.save_epoch = self._train_kwargs.get('save_epoch', 5)
        # 保存最好的训练结果
        self.best_epoch = -1

        # data set
        # 数据集参数
        # 载入邻接矩阵，(sensor_ids, sensor_id_to_ind, adj_mx)
        _, _, self._adj_mx = utils.data.load_graph_data(self._data_kwargs.get('graph_pkl_filename'))
        # 载入数据集
        self._data = utils.data.load_dataset(**self._data_kwargs)
        # self.standard_scaler = self._data['scaler']  # 标准化参数

        # 模型参数
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))  # encoder输入的维度
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder，即 history_length
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))  # encoder输出的维度，即decoder输入的维度
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))  # curriculum_learning，一种训练策略
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder，即 prediction_length

        # setup model
        # 设置模型
        self.dcrnn_model = DCRNNModel(self._adj_mx, self._logger, **self._model_kwargs).to(self.device)
        self._logger.info("Model created")
        # 构建完模型后，输出下参数数量
        self._logger.info(
            "Total trainable parameters {}".format(self.dcrnn_model.count_parameters())
        )

        # 是否需要载入模型权重文件，包含模型参数和训练状态
        self._has_saved_state = self._train_kwargs.get('has_saved_state', False)  # 是否有保存的权重文件
        self._saved_state = None  # 保存训练状态，用于训练模式 字典 ['epoch', 'optimizer_state_dict']
        self._epoch_num = -1  # 上一次训练的epoch
        if self._has_saved_state:
            # 载入模型参数，并且保存训练状态
            self.load_model()
            self._epoch_num = self._saved_state['epoch']

    def save_model(self, epoch, optimizer_state_dict, is_best=False):
        # 保存模型参数和训练状态
        save_path = self.runs_path + "models/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 若保存的是最好的训练结果，则将之前的 best_epoch 删除
        if is_best:
            save_path += "best_"
            if self.best_epoch != -1:
                os.remove(save_path + "epoch{}.pth".format(self.best_epoch))
            self.best_epoch = epoch

        config = dict(self._kwargs)

        # 保存字典 ['model_state_dict', 'epoch', 'optimizer_state_dict']
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        config['optimizer_state_dict'] = optimizer_state_dict

        # save_path = 'models/runs%d/epoch%d.pth'
        save_path += "epoch{}.pth".format(epoch)
        torch.save(config, save_path)

        self._logger.info("Saved model at {}".format(epoch))

        return save_path

    def load_model(self):
        # 载入模型权重文件
        checkpoint = torch.load(self._train_kwargs.get('model_state_pth'))

        # 保存训练状态
        self._saved_state = {'model_state_dict': checkpoint['model_state_dict'],
                             'epoch': checkpoint['epoch'],
                             'optimizer_state_dict': checkpoint['optimizer_state_dict']}

        # 载入模型参数
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        self._logger.info("Loaded model at {}".format(checkpoint['epoch']))

    def show_metrics(self, preds, labels, base_message, null_val=0.0):
        for i in range(len(preds)):
            mae = utils.metrics.masked_mae(preds[i], labels[i], null_val=null_val)
            rmse = utils.metrics.masked_rmse(preds[i], labels[i], null_val=null_val)
            mape = utils.metrics.masked_mape(preds[i], labels[i], null_val=null_val)

            message = base_message + 'horizon{:2}, MAE: {:2.4f}, RMSE: {:2.4f}, MAPE: {:2.4f}'.format(
                i + 1, mae, rmse, mape
            )

            self._logger.info(message)

        preds = torch.stack(preds, dim=0)
        labels = torch.stack(labels, dim=0)

        mae = utils.metrics.masked_mae(preds, labels, null_val=0.0)
        rmse = utils.metrics.masked_rmse(preds, labels, null_val=0.0)
        mape = utils.metrics.masked_mape(preds, labels, null_val=0.0)

        message = base_message + ' overall , MAE: {:2.4f}, RMSE: {:2.4f}, MAPE: {:2.4f}'.format(
            mae, rmse, mape
        )

        self._logger.info(message)

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val'):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            losses = []

            y_truths = []
            y_preds = []

            # 保证日志输出整洁，0.1s
            time.sleep(0.1)
            for item in tqdm(self._data['{}_loader'.format(dataset)]):
                x = item['x'].to(self.device)
                y = item['y'].to(self.device)

                # x, y = self._prepare_data(x, y)

                # 注意，这里没有将 y 和 batches_seen 作为输入
                # 因为在 val 和 test 阶段中，Decoder 都用的预测值作为输入
                output = self.dcrnn_model(x)

                loss = self._compute_loss(y, output)

                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            # TODO
            # self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            # 将输出结果返回，用于测试阶段
            # (batches, prediction_length, num_nodes, output_dim)
            y_truths = torch.cat(y_truths, dim=0)
            y_preds = torch.cat(y_preds, dim=0)

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[1]):
                # 归一化对所有的数据都要统一！！！
                # 之前我是分别对train、val、test进行不同的归一化，见dcrnn_supervisor_src.py代码
                y_truth = self._data['scaler'].inverse_transform(y_truths[:, t, :, :])
                y_pred = self._data['scaler'].inverse_transform(y_preds[:, t, :, :])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=True,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0

        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
        # 载入优化器状态
        if self._has_saved_state:
            assert 'optimizer_state_dict' in self._saved_state
            optimizer.load_state_dict(self._saved_state['optimizer_state_dict'])

        # 用于调整优化器中的学习率，注意last_epoch=self._epoch_num，若第一次训练，则为-1
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio, last_epoch=self._epoch_num)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = len(self._data['train_loader'])  # 训练集总共的batch数
        self._logger.info("a epoch has {} num_batches.".format(num_batches))

        # 已经训练的batch数，用于Scheduled-Sampling中概率p的递减
        batches_seen = num_batches * (self._epoch_num + 1)

        # epoch 上一次训练的 epoch+1 开始计数，若第一次训练，则 self._epoch_num 为 -1
        # epoch_num 范围为 0 -> 99
        for epoch_num in range(self._epoch_num + 1, epochs + 1):

            self.dcrnn_model = self.dcrnn_model.train()

            # 记录每个batch的loss，用于计算该epoch的平均loss
            losses = []

            start_time = time.time()

            # 保证日志输出整洁，0.1s
            time.sleep(0.1)
            for item in tqdm(self._data['train_loader']):
                x = item['x'].to(self.device)  # (batch_size, history_length, num_nodes, 1)
                y = item['y'].to(self.device)  # (batch_size, prediction_length, num_nodes, 1)

                optimizer.zero_grad()

                # output (batch_size, horizon, num_nodes, output_dim)
                output = self.dcrnn_model(x, y, batches_seen)

                # TODO 不知道有什么用
                # if batches_seen == 0:
                #     # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                #     optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                # 计算损失值 MAE，'train_data'表示用train的标准化参数来还原数据
                loss = self._compute_loss(y, output)

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1

                loss.backward()

                # gradient clipping - this does it in place
                # 梯度裁剪，只解决梯度爆炸问题，不解决梯度消失问题。
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()

            self._logger.info("epoch complete")

            # 更新学习率
            lr_scheduler.step()

            # 每经过一个epoch，则进行验证
            self._logger.info("evaluating now!")

            # 验证
            val_loss, val_results = self.evaluate(dataset='val')

            end_time = time.time()

            # TODO
            # self._writer.add_scalar('training loss',
            #                         np.mean(losses),
            #                         batches_seen)

            # 每经过log_every（默认1）个epoch，显示一次训练和验证的结果
            if (epoch_num + 1) % log_every == 0:
                base_message = 'Epoch [{}/{}] ({} batches_seen) '.format(epoch_num, epochs, batches_seen)
                message = 'Epoch [{}/{}] ({} batches_seen) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

                # 输出评价指标：MAE、RMSE、MAPE
                self.show_metrics(val_results['prediction'], val_results['truth'], base_message, 0.0)

            # 每经过test_every_n_epochs（默认10）个epoch，显示一次测试的结果
            if (epoch_num + 1) % test_every_n_epochs == 0:
                # 测试
                test_loss, test_results = self.evaluate(dataset='test')

                base_message = 'Epoch [{}/{}] ({} batches_seen) '.format(epoch_num, epochs, batches_seen)
                message = 'Epoch [{}/{}] ({} batches_seen) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

                # 输出评价指标：MAE、RMSE、MAPE
                self.show_metrics(test_results['prediction'], test_results['truth'], base_message, 0.0)

            # 若验证过程的loss降低了，则保存一下模型（可选）
            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num, optimizer.state_dict(), is_best=True)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
            elif val_loss >= min_val_loss:
                # 若经过50epoch，loss还没有降低，则退出训练
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

            # 每经过 save_epoch，则保存一次模型，epoch_num 从 0 开始
            if (epoch_num + 1) % self.save_epoch == 0:
                self.save_model(epoch_num, optimizer.state_dict())

    def _compute_loss(self, y_true, y_predicted):
        # 归一化对所有的数据都要统一！！！
        # 之前我是分别对train、val、test进行不同的归一化，见dcrnn_supervisor_src.py代码
        y_true = self._data['scaler'].inverse_transform(y_true)
        y_predicted = self._data['scaler'].inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
