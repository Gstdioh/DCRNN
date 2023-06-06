import csv
import pickle

import numpy as np


def get_adjacency_matrix(distance_file, sensor_ids, graph_type="direct", normalized_k=0.1):
    """
    构建邻接矩阵

    :param distance_file: data file with three columns: [from, to, distance]. distance文件
    :param sensor_ids: list of sensor ids. 传感器编号列表
    :param graph_type: 图的类型，["direct", "unDirect"]
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
                         邻接矩阵经过高斯核后，其中值小于 normalized_k 的设置为 0
    :return:
    """
    # 初始化邻接矩阵
    num_sensors = len(sensor_ids)  # N
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)  # [N, N]
    dist_mx[:] = np.inf  # 距离初始化为无穷大

    # Builds sensor id to index map.
    # 构建 sensor id 到 index 的映射 map
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    # 根据 distance 文件构建邻接矩阵
    # from, to, dist
    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)  # 读取csv文件
        for item in reader:
            row = [int(item[0]), int(item[1]), float(item[2])]
            # 不在传感器编号中，则跳过
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            # 构建有向的邻接矩阵，dist_mx[from_index][to_index] = distance
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    # 计算标准差，作为 theta
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    # 小于阈值的设为 0
    adj_mx[adj_mx < normalized_k] = 0

    # Make the adjacent matrix symmetric by taking the max.
    # 若是无向图，则构建对称矩阵
    if graph_type == "unDirect":
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    return sensor_ids, sensor_id_to_ind, adj_mx


def load_graph_data(graph_pkl_filename, **kwargs):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(graph_pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


if __name__ == '__main__':
    # PEMS04 distance文件：节点数307（编号0-306）
    sensor_ids_pems04 = [i for i in range(307)]
    sensor_ids_pems04, sensor_id_to_ind_pems04, adj_mx_pems04 = get_adjacency_matrix(
        distance_file=r"D:\1_program\0_Traffic_Predition\DCRNN_My\dataset\PEMS04\distance.csv",
        sensor_ids=sensor_ids_pems04,
        graph_type="direct",
        normalized_k=0.1)
