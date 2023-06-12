import argparse
import yaml

from utils.supervisor import Supervisor


def main(args):
    with open(args.config_filename, 'r', encoding="utf-8") as f:
        supervisor_config = yaml.safe_load(f)

        supervisor = Supervisor(**supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    # parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
