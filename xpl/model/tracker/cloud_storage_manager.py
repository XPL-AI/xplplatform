"""
This file defines the storage manager.
All interactions with loading and saving models to the files is handled here.
Also, saving and loading measurements logs are handled here as well.
"""
import os

from torch.functional import Tensor
from xpl import user
import torch
import pandas
import shutil
import datetime
import collections

import logging

from typing import Dict

from xpl.model import config

from xpl.infrastructure.storage import CloudStorageRepository

logger = logging.getLogger(__name__)


class CloudStorageManager:
    def __init__(self):

        self.__skip_jit = collections.defaultdict(lambda: False)

        self.__cloud_storage: CloudStorageRepository = CloudStorageRepository(bucket_name=config['cloud_storage_bucket'])

    def __get_model_folder_prefix(self,
                                  user_id: str,
                                  task_id: str,
                                  execution_id: str):
        return os.path.join(config.MODELS_DIR,
                            user_id,
                            task_id,
                            execution_id)

    def __get_cloud_model_folder_prefix(self,
                                        user_id: str,
                                        task_id: str,
                                        execution_id: str):
        return os.path.join('models',
                            user_id,
                            task_id,
                            execution_id)

    def save_csv_file(self,
                      user_id: str,
                      task_id: str,
                      execution_id: str,
                      dataframe: pandas.DataFrame,
                      name: str):
        if dataframe is None:
            raise BaseException('CSV file is None')
        output_file_name = os.path.join(self.__get_model_folder_prefix(user_id=user_id,
                                                                       task_id=task_id,
                                                                       execution_id=execution_id),
                                        f'{name}.csv')

        self.__makedir(output_file_name)
        logger.debug(f'{output_file_name=}')
        dataframe.to_csv(output_file_name)

        cloud_file_name = os.path.join(self.__get_cloud_model_folder_prefix(user_id=user_id,
                                                                            task_id=task_id,
                                                                            execution_id=execution_id),
                                       f'{name}.csv')

        self.__cloud_storage.upload_from_dataframe(dataframe,
                                                   blob_name=cloud_file_name)

    def load_csv_file(self,
                      user_id: str,
                      task_id: str,
                      execution_id: str,
                      name: str):
        input_file_name = os.path.join(self.__get_cloud_model_folder_prefix(user_id=user_id,
                                                                            task_id=task_id,
                                                                            execution_id=execution_id),
                                       f'{name}.csv')

        assert self.__cloud_storage.exists(input_file_name), f'{input_file_name} does not exist'
        return self.__cloud_storage.download_as_dataframe(input_file_name)

    def save_models(self,
                    models: dict,  # A dictionary of {name:pytorch_model}
                    user_id: str,
                    task_id: str,
                    execution_id: str,
                    model_id: str):
        model_path = os.path.join(self.__get_model_folder_prefix(user_id=user_id,
                                                                 task_id=task_id,
                                                                 execution_id=execution_id),
                                  model_id)

        for model_name, model in models.items():
            model_pth_path = os.path.join(model_path, f'{model_name}.pth')
            torchscript_path = os.path.join(model_path, f'{model_name}.pts')
            self.__makedir(model_pth_path)
            torch.save(model.get_state_dict(), model_pth_path)
            model_script = torch.jit.script(model)
            if model_script is not None:
                torch.jit.save(model_script, torchscript_path)

    def upload_models_to_cloud(self,
                               user_id: str,
                               task_id: str,
                               execution_id: str,
                               model_ids
                               ) -> Dict:
        saved_models = {}
        for model_id in model_ids:
            model_folder = os.path.join(self.__get_model_folder_prefix(user_id=user_id,
                                                                       task_id=task_id,
                                                                       execution_id=execution_id),
                                        model_id)
            files = [name for name in os.listdir(model_folder)]
            model_cloud_folder = os.path.join(self.__get_cloud_model_folder_prefix(user_id=user_id,
                                                                                   task_id=task_id,
                                                                                   execution_id=execution_id),
                                              model_id)
            saved_models[model_id] = {}
            for file in files:
                blob_name = os.path.join(model_cloud_folder, file)
                full_file_name = os.path.join(model_folder, file)

                url = self.__cloud_storage.upload_from_file(file_path=full_file_name, blob_name=blob_name)
                saved_models[model_id][file] = {'url': url}

        return saved_models

    def delete_all_but_these_models(self,
                                    user_id: str,
                                    task_id: str,
                                    execution_id: str,
                                    model_ids: list):
        execution_model_folder = self.__get_model_folder_prefix(user_id=user_id,
                                                                task_id=task_id,
                                                                execution_id=execution_id)
        all_models = [name
                      for name in os.listdir(execution_model_folder)
                      if os.path.isdir(os.path.join(execution_model_folder, name)) and name.startswith('model_')
                      ]
        for model_id in all_models:
            if model_id in model_ids:
                continue
            logger.debug(f'Deleting {model_id=}')
            self.__delete_model(user_id=user_id,
                                task_id=task_id,
                                execution_id=execution_id,
                                model_id=model_id)

    def __delete_model(self,
                       user_id: str,
                       task_id: str,
                       execution_id: str,
                       model_id: str):
        model_path = os.path.join(self.__get_model_folder_prefix(user_id=user_id,
                                                                 task_id=task_id,
                                                                 execution_id=execution_id),
                                  model_id)

        assert os.path.exists(model_path), model_path
        shutil.rmtree(model_path)

    def list_execution_ids(self,
                           user_id: str,
                           task_id: str
                           ) -> list[str]:

        execution_root_path = os.path.join('models',
                                           user_id,
                                           task_id)

        all_execution_ids = self.__cloud_storage.list_directories(execution_root_path,
                                                                  starts_with='execution_',
                                                                  sort_reverse=True)
        ret_execution_ids = []
        for execution_id in all_execution_ids:
            best_csv_file_path = os.path.join(self.__get_cloud_model_folder_prefix(user_id=user_id,
                                                                                   task_id=task_id,
                                                                                   execution_id=execution_id),
                                              'best.csv')
            if self.__cloud_storage.exists(best_csv_file_path):
                start_time_str = '_'.join(execution_id.split('_')[1:3])
                start_time = datetime.datetime.strptime(
                    start_time_str, '%Y-%m-%d_%H-%M-%S')
                end_time = self.__cloud_storage.get_update_time(best_csv_file_path).replace(tzinfo=None)
                execution_time = end_time - start_time
                days, hours, minutes = int(execution_time.days), int(execution_time.seconds //
                                                                     3600), int(execution_time.seconds % 3600 / 60.0)
                execution_duration = ''
                if days > 0:
                    execution_duration += f' {days=}'
                if hours > 0:
                    execution_duration += f' {hours=}'
                if minutes > 0:
                    execution_duration += f' {minutes=}'
                if execution_time.seconds < 120:
                    continue
                execution_message = f'\t(runtime: {execution_duration})'
                ret_execution_ids.append(execution_id + execution_message)

        return ret_execution_ids

    def get_last_model_id(self,
                          user_id: str,
                          task_id: str,
                          execution_id: str
                          ) -> str:
        all_model_paths = self.__get_cloud_model_folder_prefix(user_id=user_id,
                                                               task_id=task_id,
                                                               execution_id=execution_id)

        all_model_ids = self.__cloud_storage.list_directories(all_model_paths,
                                                              starts_with='model_',
                                                              sort_reverse=True)

        if len(all_model_ids) == 0:
            logger.debug(f'{execution_id=} does not have any models in it')
            return None
        return all_model_ids[0]

    def load_torchscript_model(self,
                               model_id: str):
        pass

    def load_pytorch_state_dict(self,
                                 model_name:str,
                                 user_id: str,
                                 task_id: str,
                                 execution_id: str,
                                 model_id: str
                                 ) -> dict[str, torch.Tensor]:
        model_local_directory = os.path.join(self.__get_model_folder_prefix(user_id=user_id,
                                                                            task_id=task_id,
                                                                            execution_id=execution_id),
                                             model_id)

        self.__download_model_files(user_id=user_id,
                                    task_id=task_id,
                                    execution_id=execution_id,
                                    model_id=model_id,
                                    model_local_directory=model_local_directory)

        model_path = os.path.join(model_local_directory, f'{model_name}.pth')
        if os.path.exists(model_path):
            logger.info(f'{model_name} loaded successfully')
            return torch.load(f=model_path, map_location=torch.device('cpu'))
        else:
            logger.warn(f'{model_name} does not exist')
            return {}

    def __download_model_files(self,
                               user_id: str,
                               task_id: str,
                               execution_id: str,
                               model_id: str,
                               model_local_directory: str
                               ) -> None:
        model_cloud_path = os.path.join(self.__get_cloud_model_folder_prefix(user_id=user_id,
                                                                             task_id=task_id,
                                                                             execution_id=execution_id),
                                        model_id)

        for file_full_name in self.__cloud_storage.list_blobs(model_cloud_path,
                                                              ends_with='.pth'):
            file_name = file_full_name.split('/')[-1]
            local_full_file_name = os.path.join(model_local_directory, file_name)
            if not os.path.exists(local_full_file_name):
                self.__makedir(local_full_file_name)
                self.__cloud_storage.download_to_file(blob_name=file_full_name,
                                                    destination_file_name=local_full_file_name)
        return None

    def __makedir(self,
                  file_name: str
                  ) -> None:
        os.makedirs(os.path.dirname(file_name),
                    exist_ok=True)
        return None

    def __get_list_of_subdirs(self,
                              dirname: str,
                              prefix: str = ''
                              ) -> list[str]:
        if not os.path.exists(dirname):
            return []
        all_subdirs = [name for name in os.listdir(dirname) if os.path.isdir(
            os.path.join(dirname, name)) and name.startswith(prefix)]
        return sorted(all_subdirs, reverse=True)

    def __get_list_of_files(self,
                            dirname: str,
                            postfix: str = ''
                            ) -> list[str]:
        all_files = [name for name in os.listdir(dirname) if name.endswith(postfix)]
        return sorted(all_files, reverse=True)
