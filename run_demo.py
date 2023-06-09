import argparse
import yaml

from utils.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename, 'r', encoding="utf-8") as f:
        supervisor_config = yaml.safe_load(f)

        supervisor = DCRNNSupervisor(**supervisor_config)

        # 测试
        test_loss, test_results = supervisor.evaluate(dataset='test')

        base_message = ''

        # 输出评价指标：MAE、RMSE、MAPE
        supervisor.show_metrics(test_results['prediction'], test_results['truth'], base_message, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default="configs/metr-la_demo.yaml", type=str,
                        help='Configuration filename for restoring the model.')
    # parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
