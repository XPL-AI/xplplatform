####################################################################################################
# File: dataset_service.py                                                                         #
# File Created: Wednesday, 15th September 2021 2:49:20 pm                                          #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Monday, 20th September 2021 1:38:54 pm                                            #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Dict, Optional
from repositories import DataItemRepository

from user_service import UserService
from xpl.task.task_service import TaskService
from dataset.xpl_dataloader import XPLDataLoader
import pandas

class DatasetService:
    
    def __init__(self
                 ) -> None:
        pass

    def get_data_loaders(self,
                         user_id: str,
                         task_id: str,
                         processed: bool,
                         dataset_definition: dict,
                         modality: str,
                         background_user_id: Optional[str] = None,
                         background_task_id: Optional[str] = None,
                         ) -> Dict[str, XPLDataLoader]:

        data_points: pandas.DataFrame = self.load_dataset_as_dataframe(user_id=user_id,
                                                                       task_id=task_id,
                                                                       processed=processed)
        # pretrain is a constant repository
        background_data_points = self.load_dataset_as_dataframe(user_id=background_user_id,
                                                                task_id=background_task_id,
                                                                processed=processed)

        data_loaders = {}
        for name, definition in dataset_definition.items():
            data_loaders[name] = self.get_data_loader(data_points=data_points,
                                                      background_data_points=background_data_points,
                                                      dataset_definition=definition,
                                                      modality=modality)
        return data_loaders

    def get_data_loader(self,
                        data_points: pandas.DataFrame,
                        background_data_points: Optional[pandas.DataFrame],
                        dataset_definition: dict,
                        modality: str,
                        ) -> XPLDataLoader:

        return XPLDataLoader(data_points=data_points,
                             background_data_points=background_data_points,
                             dataset_definition=dataset_definition,
                             modality=modality)

    def load_dataset_as_dataframe(self,
                                  user_id: str,
                                  task_id: str,
                                  processed: bool,
                                  concept_id: str = None,
                                  predictor_type: str = None,
                                  predictor_id: str = None,
                                  input_set: str = None
                                  ):

        table_name = self.__get_table_name(task_id,
                                           processed)

        data_item_repository = DataItemRepository(dataset_id=f'{user_id}')
        data_items = data_item_repository.select_as_data_frame(table_name=table_name,
                                                               concept_id=concept_id,
                                                               predictor_type=predictor_type,
                                                               predictor_id=predictor_id,
                                                               input_set=input_set)
        return data_items
        
    def __get_table_name(self, 
                         task_id: str, 
                         processed: bool):
        return task_id if processed is True else f'{task_id}__unprocessed'