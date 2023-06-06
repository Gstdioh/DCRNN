import os
import numpy as np
import pandas as pd
import argparse

import yaml

import utils
from utils import data
import pickle


def prepare_adj_data(sensor_ids_file, distance_file, output_pkl_file=None, normalized_k=0.1, **kwargs):
    # 构建邻接矩阵
    sensor_ids = []
    if type(sensor_ids_file) is type(1):
        # 类型是数字，则编号范围为 0 - n
        sensor_ids = [i for i in range(sensor_ids_file)]
    else:
        # 类型是字符串，则从文件中读取编号，记得将编号转换为数字
        with open(sensor_ids_file) as f:
            sensor_ids = [int(item) for item in f.read().strip().split(',')]

    _, sensor_id_to_ind, adj_mx = utils.data.get_adjacency_matrix(distance_file, sensor_ids)

    # Save to pickle file.
    # 将邻接矩阵保存到 .pkl文件
    filename = ""
    if output_pkl_file is not None:
        filename = output_pkl_file
    else:
        file = "adj_mx.pkl"
        dirpath = os.path.dirname(distance_file)
        filename = os.path.join(dirpath, file)
    with open(filename, 'wb') as f:
        print('save adj_mx file:', filename)
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)


def prepare_traffic_data(data_file, data_file_type='npz', add_time_in_day=True, add_day_in_week=False,
                         output_npz_file=None, split_list=(6, 2, 2), **kwargs):
    # 读取数据集
    data_seq = None
    if data_file_type == 'npz':
        # data_seq (times, num_nodes, 1)
        data_seq = np.load(data_file)['data'][:, :, 0: 1]

        times, num_nodes, _ = data_seq.shape

        data_list = [data_seq]

        time_one_day = 24 * 60

        # 将一天中所处的时间作为输入特征（可选，特征加 1）
        if add_time_in_day:
            # time_ind 为一天中所处的时间(0 ~ 1)，间隔为5min
            # time_ind (times)
            time_ind = [(time_now * 5 % time_one_day * 1.0) / time_one_day for time_now in range(times)]

            # time_in_day (1, num_nodes, times) -> (times, num_nodes, 1)
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))

            data_list.append(time_in_day)

        # 将所处的星期作为输入特征（可选，特征加 7）
        if add_day_in_week:
            # day_in_week 为所处的星期
            day_in_week = np.zeros(shape=(times, num_nodes, 7))
            week = 0  # 第一天是星期一
            count = 0
            for time_now in range(times):
                if count == time_one_day:
                    week = (week + 1) % 7
                    count = 0
                day_in_week[time_now, :, week] = 1
                count += 5
            data_list.append(day_in_week)

        data_seq = np.concatenate(data_list, axis=-1)
    elif data_file_type == 'h5':
        df = pd.read_hdf(data_file)

        times, num_nodes = df.shape

        # data (times, num_nodes, 1)
        data_seq = np.expand_dims(df.values, axis=-1)

        data_list = [data_seq]

        # 将一天中所处的时间作为输入特征（可选，特征加 1）
        if add_time_in_day:
            # time_ind 为一天中所处的时间(0 ~ 1)，间隔为5min
            # time_ind (times)
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")

            # time_in_day (1, num_nodes, times) -> (times, num_nodes, 1)
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))

            data_list.append(time_in_day)

        # 将所处的星期作为输入特征（可选，特征加 7）
        if add_day_in_week:
            # day_in_week 为所处的星期
            day_in_week = np.zeros(shape=(times, num_nodes, 7))
            day_in_week[np.arange(times), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        data_seq = np.concatenate(data_list, axis=-1)

    times = data_seq.shape[0]

    split_sum = sum(split_list)
    val_start = int(times * (split_list[0] * 1. / split_sum))
    test_start = int(times * ((split_list[0] + split_list[1]) * 1. / split_sum))

    # 划分数据集
    train = data_seq[:val_start, :, :]
    val = data_seq[val_start:test_start, :, :]
    test = data_seq[test_start:, :, :]

    # 保存划分的数据集
    filename = ""
    if output_npz_file is not None:
        filename = output_npz_file
    else:
        file = os.path.basename(data_file).split('.')[0]
        dirpath = os.path.dirname(data_file)
        filename = os.path.join(dirpath, "{}_{}_{}_{}".format(file, split_list[0], split_list[1], split_list[2]))
    print('save traffic data file:', filename + '.npz')
    np.savez_compressed(filename,
                        train=train,
                        val=val,
                        test=test
                        )


# 预处理数据，包括保存邻接矩阵和划分数据集
def main(args):
    with open(args.config_filename, 'r', encoding="utf-8") as f:
        # 直接用yaml.load(f)会报错
        # data_config = yaml.load(f)
        data_config = yaml.safe_load(f)

        adj_kwargs = data_config['adj_data']
        traffic_kwargs = data_config['traffic_data']

        prepare_adj_data(**adj_kwargs)
        prepare_traffic_data(**traffic_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default=r"D:\1_program\0_Traffic_Predition\DCRNN_My\configs\prepare_data.yaml",
                        type=str,
                        help='Configuration filename for preparing data.')
    args = parser.parse_args()
    main(args)

    # data = np.load(r"D:\1_program\0_Traffic_Predition\DCRNN_My\dataset\PEMS04\pems04_6_2_2.npz")
    #
    # print(data["train"].shape)
    # print(data["val"].shape)
    # print(data["test"].shape)
