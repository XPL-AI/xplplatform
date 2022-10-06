from typing import Dict, List, Optional

import pandas

from xpl.infrastructure.storage.repository import CloudStorageRepository

from xpl.user import UserService
from xpl.data.repositories import DataItemRepository, DataItem, TableExistsException


class DataService:
    def __init__(self):
        pass

    def load_dataset(self,
                     user_id,
                     task_id: str,
                     processed: bool,
                     concept_id: str = None,
                     predictor_type: str = None,
                     predictor_id: str = None,
                     input_set: str = None):
        user_service = UserService()
        user = user_service.get_user_by_id(user_id)
        """User should exist"""

        table_name = self.__get_table_name(task_id,
                                           processed)

        data_item_repository = DataItemRepository(dataset_id=f'{user.user_id}')
        data_items = data_item_repository.select(table_name=table_name,
                                                 concept_id=concept_id,
                                                 predictor_type=predictor_type,
                                                 predictor_id=predictor_id,
                                                 input_set=input_set)
        return data_items

    def load_dataset_by_row_ids(self,
                                user_id,
                                task_id: str,
                                processed: bool,
                                row_ids: List[str] = None
                                ) -> List[DataItem]:
        user_service = UserService()
        user = user_service.get_user_by_id(user_id)
        """User should exist"""

        table_name = self.__get_table_name(task_id,
                                           processed)

        data_item_repository = DataItemRepository(dataset_id=f'{user.user_id}')
        data_items = data_item_repository.select(table_name=table_name, row_ids=row_ids)
        return data_items

    def ingest_data_items(self,
                          user_id,
                          task_id,
                          processed: bool,
                          data_items: list[DataItem]):
        table_name = self.__get_table_name(task_id, processed)

        __dataItemRepository = DataItemRepository(dataset_id=f'{user_id}')
        __dataItemRepository.insert(data_items, table_name=table_name)

    def ingest_data_items_from_data_frame(self,
                                          user_id,
                                          task_id,
                                          processed: bool,
                                          data_frame: pandas.DataFrame):
        """
            Inserts a DataFrame of data items
            Dataframe should have structure:

        """

        table_name = self.__get_table_name(task_id,
                                           processed)

        __dataItemRepository = DataItemRepository(dataset_id=f'{user_id}')
        __dataItemRepository.insert_from_data_frame(data_frame,
                                                    table_name=table_name)

    def setup_dataset(self,
                      user_id,
                      task_id,
                      modality):
        """
        @param user_id: An Id of the user who owns experiment and associated data.
        @param task_id: An id of the experiment.
        @param modality: The modality of the data.
        @return:
        """

        self.__provision_datasets_storage_for_user(user_id)

        data_item_repository = DataItemRepository(dataset_id=f'{user_id}')

        unprocessed_table_name = self.__get_table_name(task_id,
                                                       processed=False)
        processed_table_name = self.__get_table_name(task_id,
                                                     processed=True)

        try:
            data_item_repository.create_table(table_name=unprocessed_table_name,
                                              dataset_id=f'{user_id}',
                                              modality_type=modality,
                                              include_text=True,
                                              include_value=True)
        except TableExistsException:
            raise TaskExistsException(f'Task "{unprocessed_table_name=}", {user_id=} was already setup.')

        try:
            data_item_repository.create_table(table_name=processed_table_name, dataset_id=f'{user_id}', modality_type=modality)
        except TableExistsException:
            raise TaskExistsException(f'Task "{processed_table_name=}", {user_id=} was already setup.')

    def __get_table_name(self,
                         task_id,
                         processed):
        return task_id if processed is True else f'{task_id}__unprocessed'

    def __provision_datasets_storage_for_user(self,
                                              user_id):
        """
        Creates a default Google Cloud Storage bucket for the user's datasets.
        @param user_id: an Id of the user in XPL platform
        """
        user_service = UserService()
        user = user_service.get_user_by_id(user_id)

        if user.datasets_storage_bucket is None:
            default_location = 'EUROPE-NORTH1'
            user_datasets_bucket_name = f'xplai-datasets-{user.user_id}-{default_location}'.lower()
            storage = CloudStorageRepository()
            if not storage.bucket_exist(user_datasets_bucket_name):
                storage.create_bucket(bucket_name=user_datasets_bucket_name,
                                      location=default_location)

            if user.email != 'stub_user':
                user_service.set_user_datasets_storage_bucket(user_id=user_id,
                                                              datasets_storage_bucket=user_datasets_bucket_name)


class DataServiceException(Exception):
    """Basic DataService exception"""
    pass


class TaskExistsException(Exception):
    """Raised when attempt to setup task with the name that is already taken"""
    pass
