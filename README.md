# DCRNN

## 数据集下载

数据集：`dataset/{METR-LA, PEMS-BAY, PEMS04}`，下载后放在项目下即可。

链接：https://pan.baidu.com/s/11nEC0EivmoKjZLG8LmWgYg?pwd=rqi5 
提取码：rqi5

## 预处理邻接矩阵和数据集

运行以下命令，如：在路径 `dataset/PEMS04` 下生成文件 `adj_mx.pkl` 和 `pems04_6_2_2.npz`，
分别代表预处理的邻接矩阵和数据集。

```shell script
# pems04
python prepare_data.py --config_filename configs/prepare_data_pems04.yaml

# metr-la
python prepare_data.py --config_filename configs/prepare_data_metr-la.yaml
```

`python prepare_data.py --config_filename configs/prepare_data_pems04.yaml`

## 训练DCRNN模型（单精度 float）

运行以下命令，即可开始进行训练，如：具体配置可以查看文件 `configs/dcrnn_pems04.yaml`

```shell script
# pems04
python dcrnn_train.py --config_filename configs/dcrnn_pems04.yaml

# metr-la
python dcrnn_train.py --config_filename configs/dcrnn_metr-la.yaml
```

如：权重文件和日志保存在 `runs/PEMS04/train` 中。
