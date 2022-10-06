####################################################################################################
# File: model_service.py                                                                           #
# File Created: Monday, 7th June 2021 12:58:36 pm                                                  #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Thursday, 11th November 2021 1:56:59 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


from xpl import user
from xpl.model.neural_net.xpl_model import XPLModel
from xpl.model.tracker.model_monitor import ModelMonitor
from typing import Optional, Dict
from xpl.model.model_factory import ModelFactory
import logging

logger = logging.getLogger(__name__)


class ModelService:

    def __init__(self):

        self.__model_monitor = ModelMonitor()
        self.__model_factory = ModelFactory()

        logger.info("____ new MMU created ____")

    def get_models(self,
                   user_id: str,
                   task_id: str,
                   background_user_id: str,
                   background_task_id: str,
                   model_definitions: dict,
                   ) -> dict:

        models = {}
        for model_name, model_definition in model_definitions.items():
            model = self.__model_factory.generate_model(name=model_name,
                                                        definition=model_definition)
            self.__model_monitor.resume(model=model,
                                        model_name=model_name,
                                        user_id=user_id,
                                        task_id=task_id,
                                        background_user_id=background_user_id,
                                        background_task_id=background_task_id,
                                        model_size=model_definitions[model_name]['model_size'],
                                        modality=model_definitions[model_name]['modality']
                                        )
            models[model_name] = model

        return models

    def log_batch_measurements(self,
                               input_set: str,
                               graph_name: str,
                               measurements: dict
                               ) -> None:
        self.__model_monitor.log_batch_measurements(input_set=input_set,
                                                    graph_name=graph_name,
                                                    measurements=measurements)

    def end_cycle(self,
                  user_id:str,
                  task_id:str,
                  input_set,
                  graph_name
                  ) -> Dict:
        return self.__model_monitor.end_cycle(user_id=user_id,
                                              task_id=task_id,
                                              input_set=input_set,
                                              graph_name=graph_name)

    def save_pretrained_model(self,
                              user_id: str,
                              task_id: str,
                              models: dict[str, XPLModel],
                              modality: str,
                              model_size: str,
                              ) -> None:
        self.__model_monitor.save_pretrained_model(user_id=user_id,
                                                   task_id=task_id,
                                                   models=models,
                                                   modality=modality,
                                                   model_size=model_size)
