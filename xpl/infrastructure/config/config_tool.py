import argparse

from typing import Dict

from xpl.infrastructure.config import ConfigService, get_config, list_configs, print_config_dict, deploy_from_local_file
from xpl.infrastructure.config import utils


OFFSET = '  '


def main():
    parser = argparse.ArgumentParser(description='XPL Platform config tool',
                                     add_help=False)
    parser.add_argument("command")
    parser.add_argument('-e', '--env')
    parser.add_argument('-h', '--host')
    parser.add_argument('-c', '--config')

    args = parser.parse_args()

    if args.command.lower() == 'list':
        env = input("Specify environment :") if args.env is None else args.env
        configs = list_configs(env=env.upper()).values()
        configs = sorted(configs, key=lambda k: k['id'])

        utils.print_table(items=configs, columns=['id', 'local', 'deployed'])
    if args.command.lower() == 'describe':
        config_name = input("Specify config_name :") if args.config is None else args.config
        config = get_config(config_name=config_name)
        print_config_dict(config)
    if args.command.lower() == 'deploy':
        config_name = input("Specify config_name :") if args.config is None else args.config
        decision = input(f"{config_name} config will be deployed. Deploy? (Y, n) :")
        if decision == 'Y':
            deploy_from_local_file(config_name)
            print(f"{config_name} deployed")
        else:
            print(f"Deploy cancelled")

    else:
        print('Unknown command')


if __name__ == '__main__':
    main()