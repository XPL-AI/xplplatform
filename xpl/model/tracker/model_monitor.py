import logging

from typing import Dict

from torch.functional import Tensor
from xpl.model.neural_net.xpl_model import XPLModel

from graphical_user_interface import GraphicalUserInterface
from xpl.user.text_based_user_interface import TextBasedUserInterface
from xpl.model.tracker.id_tracker import IDTracker
from xpl.model.tracker.cloud_storage_manager import CloudStorageManager
from xpl.model.tracker.measurement_tracker import MeasurementTracker
# from xpl.user.graphical_user_interface import GraphicalUserInterface

logger = logging.getLogger(__name__)


class ModelMonitor:

    def __init__(self):
        self.__models = {}
        self.__resume_model_state_dicts = None
        self.__commit_iteration = None
        self.__first_training_step = True
        self.__iteration = 0

        self.__id_tracker = IDTracker()
        self.__measurement_tracker = MeasurementTracker()

        self.__start_time = self.__id_tracker.get_start_time()

        self.__storage_manager = CloudStorageManager()
        self.__text_based_user_interface = TextBasedUserInterface()
        self.__graphical_user_interface = GraphicalUserInterface()

    def log_batch_measurements(self,
                               input_set: str,
                               graph_name: str,
                               measurements: dict
                               ) -> None:

        if input_set == 'training':
            self.__iteration += 1
            input_set = 'train'

            if self.__first_training_step == True:
                self.__first_training_step = False
                self.__id_tracker.generate_new_model_id()

        model_id = self.__id_tracker.get_current_model_id()
        self.__measurement_tracker.log_batch_measurements(iteration=self.__iteration,
                                                          input_set=input_set,
                                                          graph_name=graph_name,
                                                          measurements=measurements,
                                                          model_id=model_id)
        self.__text_based_user_interface.display_results_on_terminal(self.__iteration,
                                                                     self.__measurement_tracker.get_display())

    def end_cycle(self,
                  user_id: str,
                  task_id: str,
                  input_set: str,
                  graph_name: str,
                  ) -> Dict:
        if input_set == 'training':
            input_set = 'train'

        self.__first_training_step = True
        model_id = self.__id_tracker.get_current_model_id()
        self.__measurement_tracker.end_iteration_cycle(input_set=input_set,
                                                       graph_name=graph_name,
                                                       model_id=model_id)

        self.__text_based_user_interface.display_results_on_terminal(self.__iteration,
                                                                     self.__measurement_tracker.get_display())
        self.__graphical_user_interface.plot_graph(epoch_results=self.__measurement_tracker.get_epoch_results(),
                                                   best_results=self.__measurement_tracker.get_best_results())
        return self.__commit(user_id=user_id,
                             task_id=task_id)

    def resume(self,
               model: XPLModel,
               model_name: str,
               user_id: str,
               task_id: str,
               background_user_id: str,
               background_task_id: str,
               modality: str,
               model_size: str):

        self.__models[model_name] = model

        all_execution_ids = self.__storage_manager.list_execution_ids(user_id=user_id,
                                                                      task_id=task_id)
        execution_id = self.__text_based_user_interface.ask_user_for_execution_id(all_execution_ids)

        if execution_id is None:
            logger.info(f'resuming from pretrained model')
            self.__load_model(model=model,
                              model_name=model_name,
                              user_id=background_user_id,
                              task_id=background_task_id,
                              execution_id=modality,
                              model_id=model_size)

        else:
            model_id = self.__storage_manager.get_last_model_id(user_id=user_id,
                                                                task_id=task_id,
                                                                execution_id=execution_id)
            logger.info(f'resuming from {model_id=}')
            self.__load_model(model=model,
                              model_name=model_name,
                              user_id=user_id,
                              task_id=task_id,
                              execution_id=execution_id,
                              model_id=model_id)

    def __load_model(self,
                     model: XPLModel,
                     model_name: str,
                     user_id: str,
                     task_id: str,
                     execution_id: str,
                     model_id: str):
        assert model_name in self.__models.keys()

        loaded_state_dict = \
            self.__storage_manager.load_pytorch_state_dict(model_name=model_name,
                                                           user_id=user_id,
                                                           task_id=task_id,
                                                           execution_id=execution_id,
                                                           model_id=model_id)
        model.load_state_dict(loaded_state_dict)

    def __commit(self,
                 user_id: str,
                 task_id: str,
                 ) -> Dict:
        """
        @return: a dictionary of best performing models that were preserved.
        """
        execution_id = self.__id_tracker.get_execution_id()
        model_id = self.__id_tracker.get_current_model_id()

        measurement_dataframe = self.__measurement_tracker.get_epoch_results()
        best_dataframe = self.__measurement_tracker.get_best_results()

        logger.debug(f'{best_dataframe=}')

        self.__storage_manager.save_csv_file(user_id=user_id,
                                             task_id=task_id,
                                             execution_id=execution_id,
                                             dataframe=measurement_dataframe,
                                             name='measurements')

        self.__storage_manager.save_csv_file(user_id=user_id,
                                             task_id=task_id,
                                             execution_id=execution_id,
                                             dataframe=best_dataframe,
                                             name='best')

        list_of_preserved_model_ids = [model_id]
        if 'model_id' in best_dataframe.columns:
            list_of_preserved_model_ids = list(
                set(best_dataframe['model_id'].to_list() + [model_id]))

        logger.info(f'{list_of_preserved_model_ids=}')
        if self.__commit_iteration == self.__iteration:
            logger.debug(f'==> already committed at {self.__commit_iteration=}')
        else:
            self.__commit_iteration = self.__iteration
            self.__storage_manager.save_models(models=self.__models,
                                               user_id=user_id,
                                               task_id=task_id,
                                               execution_id=execution_id,
                                               model_id=model_id)

        self.__storage_manager.delete_all_but_these_models(user_id=user_id,
                                                           task_id=task_id,
                                                           execution_id=execution_id,
                                                           model_ids=list_of_preserved_model_ids)
        saved_models = self.__storage_manager.upload_models_to_cloud(user_id=user_id,
                                                                     task_id=task_id,
                                                                     execution_id=execution_id,
                                                                     model_ids=list_of_preserved_model_ids)

        return saved_models

    def save_pretrained_model(self,
                              user_id: str,
                              task_id: str,
                              models: dict[str, XPLModel],
                              modality: str,
                              model_size: str,
                              ) -> None:
        self.__storage_manager.save_models(models=models,
                                           user_id=user_id,
                                           task_id=task_id,
                                           execution_id=modality,
                                           model_id=model_size)
        self.__storage_manager.upload_models_to_cloud(user_id=user_id,
                                                      task_id=task_id,
                                                      execution_id=modality,
                                                      model_ids=[model_size])
