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
python prepare_data.py --config_filename configs/PEMS04/pems04_prepare.yaml

# metr-la
python prepare_data.py --config_filename configs/METR-LA/metr-la_prepare.yaml
```

## 训练模型（单精度 float）

运行以下命令，即可开始进行训练，如：具体配置可以查看文件 `configs/PEMS04/pems04_train.yaml`

```shell script
# pems04
python dcrnn_train.py --config_filename configs/PEMS04/pems04_train.yaml

# metr-la
python dcrnn_train.py --config_filename configs/METR-LA/metr-la_train.yaml
```

如：权重文件和日志保存在 `runs/PEMS04/train` 中。

## 测试模型

修改配置文件中的两处，`model_state_pth`：要测试的模型权重文件路径，`has_saved_state`：true

运行以下命令，即可开始进行测试，如：具体配置可以查看文件 `configs/PEMS04/pems04_demo.yaml`

```shell script
# pems04
python run_demo.py --config_filename configs/PEMS04/pems04_demo.yaml

# metr-la
python run_demo.py --config_filename configs/METR-LA/metr-la_demo.yaml
```
