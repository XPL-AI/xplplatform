import os
import json

from typing import Dict

from google.cloud import firestore


def reload_config():
    config = get_config(config_name=f'{ConfigService.ENV}.{ConfigService.HOST}')
    ConfigService.CONFIG = config


def get_config(config_name: str):
    # Load configuration from Firestore
    config_document_snapshot = \
        ConfigService.FIRESTORE_CLIENT.collection('configs').document(config_name).get()

    config = None
    if config_document_snapshot.exists:
        config = config_document_snapshot.to_dict()

    # Load configuration from local file and override config loaded from Firestore.
    config_filename = f'{ConfigService.CONFIGS_DIR}/{config_name}.json'
    if os.path.exists(config_filename):
        config = json.loads(open(config_filename, 'r').read())
    elif config:
        # If local config file doesn't exist - create one from config loaded from Firestore.
        with open(file=config_filename, mode='w') as config_file:
            config_file.write(json.dumps(config, indent=4))

    if config is None:
        raise Exception(f'Failed to load config: HOST={ConfigService.HOST} ENV={ConfigService.ENV}\n'
                        f'Neither local file={config_filename} nor Firestore document={config_name} were found')

    return config


def list_configs(env: str):
    env = env.upper()
    documents_stream = ConfigService.FIRESTORE_CLIENT.collection('configs').stream()
    configs = {}
    for doc in documents_stream:
        if doc.id.upper().startswith(env):
            configs[doc.id] = {'id': doc.id, 'deployed': 'True', 'local': 'False'}
    local_configs = [f.replace('.json', '') for f in os.listdir(ConfigService.CONFIGS_DIR)
                     if os.path.isfile(os.path.join(ConfigService.CONFIGS_DIR, f)) and f.startswith(env)]

    for local_config in local_configs:
        if local_config in configs:
            configs[local_config]['local'] = 'True'
        else:
            configs[local_config] = {'id': local_config, 'local': 'True', 'deployed': 'False'}
    return configs


def deploy_from_local_file(config_name: str):
    config: dict
    config_filename = f'{ConfigService.CONFIGS_DIR}/{config_name}.json'
    if os.path.exists(config_filename):
        with open(config_filename, 'r') as file:
            config = json.loads(file.read())
            ConfigService.FIRESTORE_CLIENT \
                .collection('configs') \
                .document(config_name) \
                .set(config)
    else:
        raise Exception(f'Config file {config_filename} does ot exist')


def print_loaded_config():
    print(f'ConfigService.CONFIGS_DIR: {ConfigService.CONFIGS_DIR}')
    print(f'ConfigService.ENV: {ConfigService.ENV}')
    print(f'ConfigService.HOST: {ConfigService.HOST}')
    print_config_dict(ConfigService.CONFIG)


def print_config_dict(config: Dict):
    offset = '  '
    for unit_key, unit_config in config.items():
        print("\nUNIT:", unit_key)
        for key in unit_config:
            print(f'{offset}{key}: {unit_config[key]}')


class ConfigService:
    def __init__(self, unit):
        self.__unit = unit
        if unit not in ConfigService.CONFIG:
            raise ConfigException(f'Unit config section absent in the loaded config: '
                                  f'UNIT={unit} HOST={ConfigService.HOST} ENV={ConfigService.ENV}')

    def __getitem__(self, config_parameter):
        if config_parameter in ConfigService.CONFIG[self.__unit]:
            return ConfigService.CONFIG[self.__unit][config_parameter]
        else:
            raise ConfigException(f'Config parameter absent in the loaded config: '
                                  f'PARAMETER={config_parameter} UNIT={self.__unit} HOST={ConfigService.HOST} ENV={ConfigService.ENV}')

    def __contains__(self, config_parameter):
        return config_parameter in ConfigService.CONFIG[self.__unit]

    XPL_CODE_DIR = os.environ['XPL_CODE_DIR']
    CONFIGS_DIR = os.path.join(os.environ['XPL_CODE_DIR'], 'configs')
    MODELS_DIR = os.path.join(os.environ['XPL_CODE_DIR'], 'models')
    DATA_DIR = os.path.join(os.environ['XPL_CODE_DIR'], 'data')
    ENV = os.environ['XPL_ENV']
    HOST: str = os.environ['XPL_HOST']
    CONFIG: dict = None

    FIRESTORE_CLIENT: firestore.Client = firestore.Client()
    FIRESTORE_COLLECTION = 'configs'


class ConfigException(Exception):
    pass


if ConfigService.CONFIG is None:
    # print("config loaded")
    reload_config()
