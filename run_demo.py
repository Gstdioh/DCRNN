import argparse
import yaml
import numpy as np

from utils.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename, 'r', encoding="utf-8") as f:
        supervisor_config = yaml.safe_load(f)

        supervisor = DCRNNSupervisor(**supervisor_config)
        mean_score, outputs = supervisor.evaluate('test')
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))

        supervisor.train()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
