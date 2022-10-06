"""
Filename: /data/git/xplplatform/xpl/tasks/training_task.py
Path: /data/git/xplplatform/xpl/tasks
Created Date: Thursday, May 6th 2021, 2:30:29 pm
Author: Ali S. Razavian

Copyright (c) 2021 XPL Technologies AB
"""
from typing import Dict

from google.api_core import exceptions
from xpl.user import UserService

from xpl.task.task_service import TaskService
from xpl.task.entities import Task, Model, ModelComponent

from xpl.dataset.dataset_service import DatasetService
from xpl.dataset.dataset.xpl_dataloader import XPLDataLoader

from xpl.train.train_service import TrainService

from xpl.model.model_service import ModelService
from xpl.model.neural_net.xpl_model import XPLModel

from xpl.measurer.measurer_service import measurerService
from xpl.measurer.xpl_measurer import XPLMeasurer

from xpl.train.procedure import TrainingProcedure


class TrainingCoordinator:

    def __init__(self,
                 user_name: str,
                 task_name: str,
                 few_shot_learning: bool):

        train_service = TrainService()
        model_service = ModelService()
        data_service = DatasetService()
        measurer_service = measurerService()

        self.__task = self.resolve_user_name_and_task_name(user_name=user_name,
                                                           task_name=task_name)
        self.__user_id = self.__task.user_id
        self.__task_id = self.__task.task_id
        self.__task_type = 'recognition'
        self.__task_name = self.__task.name
        self.__model_size = self.__task.model_size
        self.__concepts = self.__task.concepts
        self.__modality = self.__task.modality

        self.__background_task = self.resolve_user_name_and_task_name(user_name='pretrain@xpl.ai',
                                                                      task_name=f'{self.__task.modality}_background')

        self.__background_user_id = self.__background_task.user_id
        self.__background_task_id = self.__background_task.task_id

        # First we initialize the training service.
        # information about the training pipeline are defined here

        experiment_definition: Dict = train_service.get_experiment_definition(task_name=self.__task_name,
                                                                              task_type=self.__task_type,
                                                                              model_size=self.__model_size,
                                                                              concepts=self.__concepts,
                                                                              modality=self.__modality)

        self.__measurers: Dict[str, XPLMeasurer] = measurer_service.get_measurers(definitions=experiment_definition['measurers'])

        models: Dict[str, XPLModel] = model_service.get_models(user_id=self.__user_id,
                                                               task_id=self.__task_id,
                                                               background_user_id=self.__background_user_id,
                                                               background_task_id=self.__background_task_id,
                                                               model_definitions=experiment_definition['models'],
                                                               )
        
        self.__models = models

        if few_shot_learning:
            select_from_processed_dataset = False
        else:
            select_from_processed_dataset = True

        data_loaders: Dict[str, XPLDataLoader] = data_service.get_data_loaders(user_id=self.__user_id,
                                                                               task_id=self.__task_id,
                                                                               processed=select_from_processed_dataset,
                                                                               dataset_definition=experiment_definition['data_loaders'],
                                                                               modality=self.__modality,
                                                                               background_user_id=self.__background_user_id,
                                                                               background_task_id=self.__background_task_id,
                                                                               )

        dags = train_service.get_dags(definitions=experiment_definition['dags'],
                                      models=self.__models,
                                      measurers=self.__measurers,
                                      data_loaders=data_loaders)

        self.__procedure = TrainingProcedure(user_id=self.__user_id,
                                             task_id=self.__task_id,
                                             models=self.__models,
                                             dags=dags,
                                             model_service=model_service,
                                             few_shot_learning=few_shot_learning)

    def start_training(self):
        """Train indefinitely until user interrupt"""
        while True:
            epoch_results = self.__procedure.train_for_one_epoch()
            if epoch_results.best_performing_model:
                """At the end of every epoch update model associated with the task."""
                if self.__task.model:
                    version = self.__task.model.version + 1
                else:
                    version = 1

                model = Model(
                    model_id=epoch_results.best_performing_model_id,
                    components={},
                    output={},
                    version=version
                )

                for _, m in self.__models.items():
                    if 'output_channel_names' in m.definition:
                        for idx, output_channel in enumerate(m.definition['output_channel_names']):
                            model.output[str(idx)] = output_channel

                for key, component in epoch_results.best_performing_model.items():
                    if key.endswith('.pts') or key.endswith('.ptl'):
                        model.components[key] = ModelComponent(url=component['url'],
                                                               name=key)

                attempt_count = 0
                while attempt_count < 3:
                    try:
                        attempt_count += 1
                        task = TaskService().update_task_active_model(task_id=self.__task.task_id,
                                                                      model=model)
                        self.__task = task
                        break
                    except exceptions.Unknown as e:
                        print(e)
                        continue

    def resolve_user_name_and_task_name(self,
                                        user_name: str,
                                        task_name: str
                                        ) -> Task:
        task_service = TaskService()
        user_service = UserService()
        user = user_service.resolve_user(username=user_name)
        user_tasks = task_service.list_user_tasks(user_id=user.user_id)
        tasks_matching_name = []
        for task in user_tasks:
            if task.name and task.name.lower().strip() == task_name:
                tasks_matching_name.append(task)

        assert len(tasks_matching_name) == 1, f'Fatal error, \n{tasks_matching_name=}\n is not unique for \n{user=}'

        task = tasks_matching_name[0]
        return task
