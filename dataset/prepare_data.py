import os
import numpy as np
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
        # 类型是字符串，则从文件中读取编号
        with open(sensor_ids_file) as f:
            sensor_ids = f.read().strip().split(',')

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


def prepare_traffic_data(data_file, output_npz_file=None, split_list=(6, 2, 2), **kwargs):
    # 读取数据集
    data_seq = np.load(data_file)['data']  # (sequence_length, num_of_vertices, num_of_features)
    sequence_length = data_seq.shape[0]

    split_sum = sum(split_list)
    val_start = int(sequence_length * (split_list[0] * 1. / split_sum))
    test_start = int(sequence_length * ((split_list[0] + split_list[1]) * 1. / split_sum))

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
    print('save traffic data file:', filename)
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
