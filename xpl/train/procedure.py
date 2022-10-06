import time
import torch

from pydantic import BaseModel
from typing import Optional, Tuple, Dict

from xpl.train.dag import DAG

from xpl.model.model_service import ModelService
from xpl.model.neural_net.xpl_model import XPLModel


class TrainBatchResult(BaseModel):
    epoch_complete: bool
    best_performing_model_id: Optional[str]
    best_performing_model: Optional[dict]
    measurements: Optional[dict]


class TrainEpochResult(BaseModel):
    best_performing_model_id: str
    best_performing_model: Optional[dict]
    measurements: Optional[dict]


class TrainingProcedure:
    def __init__(self,
                 user_id: str,
                 task_id: str,
                 models: dict[str, XPLModel],
                 dags: dict[str, DAG],
                 model_service: ModelService,
                 snapshot_minutes: int = 60,
                 few_shot_learning: bool = False,
                 ):
        self.__user_id = user_id
        self.__task_id = task_id
        self.__dags = dags
        self.__models = models
        self.__last_backup_time = time.time()
        self.__model_service = model_service
        self.__snapshot_period = snapshot_minutes * 60
        self.__few_shot_learning = few_shot_learning

    def train_for_one_batch(self,
                            step: bool = True,
                            ) -> TrainBatchResult:

        [model.train() for model in self.__models.values()]

        for k in self.__models.keys():
            if k in ['image_rep'] and self.__few_shot_learning:
                self.__models[k].eval()

        measurements = {}
        epoch_complete = False
        best_model = None
        bets_model_id = None
        for dag_name, dag in self.__dags.items():
            measurement = dag.execute(input_set='training')

            if not not measurement: # measurement is not empty
                assert 'loss' in measurement.keys(), f'{measurement.keys()=} must contain "loss"'
                self.__model_service.log_batch_measurements(input_set='training',
                                                            graph_name=dag_name,
                                                            measurements=measurement, )
                self.backward(measurement['loss'])
                measurements[dag_name] = measurement
            else:
                self.__model_service.end_cycle(user_id=self.__user_id,
                                               task_id=self.__task_id,
                                               input_set='training',
                                               graph_name=dag_name)
                bets_model_id, best_model = self.eval_for_one_epoch(dag_name)
                epoch_complete = True

            # Sometimes, tasks take hours or days to finish an epoch.
            # I take a snapshop once an hour to make sure we can visualize the  performance
            if (time.time() - self.__last_backup_time) > self.__snapshot_period:
                for dag_name in self.__dags.keys():
                    self.__model_service.end_cycle(user_id=self.__user_id,
                                                   task_id=self.__task_id,
                                                   input_set='train',
                                                   graph_name=dag_name)
                    self.__last_backup_time = time.time()

        if step:
            self.step()

        return TrainBatchResult(epoch_complete=epoch_complete,
                                best_performing_model_id=bets_model_id,
                                best_performing_model=best_model,
                                measurements=measurements)

    def train_for_one_epoch(self,
                            snapshot_minutes: int = 60  # snapshot models every 60 minutes
                            ) -> TrainEpochResult:
        """
        train the models for one epoch over the entire dataset

        Args:
            snapshot_minutes (int, optional): upperbound on snapshot interval in minutes.
            Defaults to 60 minutes, meaning we snapshot at least once an hour.

        Returns:
            None
        """

        while True:
            batch_train_result = self.train_for_one_batch()
            if batch_train_result.epoch_complete is True:
                return TrainEpochResult(best_performing_model_id=batch_train_result.best_performing_model_id,
                                        best_performing_model=batch_train_result.best_performing_model,
                                        measurements=batch_train_result.measurements)

    def eval_for_one_epoch(self, dag_name) -> Tuple[str, Dict]:
        """evaluate the model for one epoch

        Returns:
            None
        """
        [model.eval() for model in self.__models.values()]
        dag = self.__dags[dag_name]

        all_input_sets = dag.get_input_sets()
        with torch.no_grad():
            for input_set in all_input_sets:
                if input_set in ['training', 'train']:
                    continue
                while True:
                    measurement = dag.execute(input_set=input_set)
                    if not not measurement:
                        assert 'loss' in measurement.keys(), f'{measurement.keys()=} must contain "loss"'
                        self.__model_service.log_batch_measurements(input_set=input_set,
                                                                    graph_name=dag_name,
                                                                    measurements=measurement)
                    else:
                        best_models = self.__model_service.end_cycle(user_id=self.__user_id,
                                                                     task_id=self.__task_id,
                                                                     input_set=input_set,
                                                                     graph_name=dag_name)
                        bets_model_id, best_model = next(iter(best_models.items()))
                        return bets_model_id, best_model

        return None

    def backward(self,
                 losses: dict,
                 ) -> None:
        """
        backpropagates the loss in the network of models.
        pytorch keeps track of the graph of backpropagated

        Args:
            losses (dict): [description]
        """
        total_loss = sum(loss.mean() for loss in losses.values())
        total_loss.backward()

    def step(self,
             ) -> None:
        """
        Updates the neural network's parameters
        """

        [model.step() for model in self.__models.values()]
        [model.zero_grad() for model in self.__models.values()]
        return None
