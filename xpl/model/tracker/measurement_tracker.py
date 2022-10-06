"""
Filename: /data/git/xplplatform/xpl/model/measurement_monitor.py
Path: /data/git/xplplatform/xpl/model
Created Date: Monday, May 3rd 2021, 9:36:07 am
Author: Ali S. Razavian

Copyright (c) 2021 XPL Technologies AB
"""

from collections import defaultdict
import torch
import pandas
import datetime

import logging


logger = logging.getLogger(__name__)


class MeasurementTracker:
    def __init__(self):
        self.__all_results = {}
        self.__epoch_results = pandas.DataFrame({})
        self.__last_epoch_results = pandas.DataFrame({})
        self.__lower_is_better = {}
        self.__best_results = {}
        self.__parse_metrics()
        self.__measurements_to_be_displayed = pandas.DataFrame({})
        self.__ignore_measurements = []

    def ignore_measurements(self, 
                            measurement_list: list
                            ) -> None:
        self.__ignore_measurements += measurement_list
        return None

    def get_best_results(self
                         )-> pandas.DataFrame:
        return pandas.DataFrame(self.__best_results).T

    def get_epoch_results(self
                          )-> pandas.DataFrame:
        return self.__epoch_results

    def get_last_epoch_results(self
                               ) -> pandas.DataFrame:
        return self.__last_epoch_results

    def set_epoch_results(self, 
                          epoch_results
                          ) -> None:
        self.__epoch_results = epoch_results
        self.__update_best_models()
        return None

    def get_display(self
                    ) -> pandas.DataFrame:
        return self.__measurements_to_be_displayed

    def __update_display_measurements(self, 
                                      measurements: dict[str, float]
                                      ) -> None:

        for k, v in measurements.items():
            if k == 'iteration':
                continue
            metric, objective = k.split('.')
            self.__measurements_to_be_displayed.loc[objective, f'{metric}.last_batch'] = v

        if len(self.__best_results) > 0:
            for k, v in self.__best_results.items():
                metric, objective, _ = k.split('.')
                self.__measurements_to_be_displayed.loc[objective, f'{metric}.best'] = v['measurements']
                self.__measurements_to_be_displayed.loc[objective, f'{metric}.best.iter'] = int(v['iteration'])

        self.__measurements_to_be_displayed.sort_index(inplace=True)
        return None

    def __update_display_summary(self, 
                                 current_cycle_summary: dict
                                 ) -> None:
        for k, v in current_cycle_summary.items():
            metric, objective, _ = k.split('.')
            (_, _, input_set, _), value = list(v.items())[0]
            self.__measurements_to_be_displayed.loc[objective, f'{metric}.last_{input_set}'] = value
        return None

    def log_batch_measurements(self,
                               iteration: int,
                               input_set: str,
                               graph_name: str,
                               measurements: dict,
                               model_id: str
                               ) -> None:
        measurements = self.__measurements_from_pytorch(measurements)
        measurements['iteration'] = iteration
        self.__update_display_measurements(measurements)
        timestamp = datetime.datetime.utcnow().timestamp()
        self.__all_results[(model_id, graph_name, input_set,
                            iteration, timestamp)] = measurements
        return None

    def end_iteration_cycle(self,
                            graph_name: str,
                            input_set: str,
                            model_id: str,
                            ) -> None:

        current_epoch_results = {k: v for k, v in self.__all_results.items()
                                 if k[1] == graph_name and k[2] == input_set}

        self.__all_results = {k: v for k, v in self.__all_results.items()
                              if k[1] != graph_name or k[2] != input_set}

        if len(current_epoch_results) == 0:
            return

        current_cycle_summary = pandas.DataFrame(current_epoch_results).T
        current_cycle_summary.index.names = ['model_id', 'graph_name', 'input_set', '_', 'timestamp']
        new_column_names = {k: f'{k}.{graph_name}' for k in current_cycle_summary.columns if k != 'iteration'}
        current_cycle_summary.rename(columns=new_column_names, inplace=True)

        logger.debug(f'{current_cycle_summary=}')
        current_cycle_summary = current_cycle_summary.mean(level=['model_id', 'graph_name', 'input_set'])

        current_cycle_summary.reset_index(inplace=True)
        current_cycle_summary.astype({'iteration': 'int32'})
        current_cycle_summary.set_index(['model_id', 'graph_name', 'input_set', 'iteration'], inplace=True)

        self.__update_display_summary(current_cycle_summary.to_dict())
        self.__last_epoch_results = current_cycle_summary

        self.__epoch_results = pandas.concat(
            [self.__epoch_results, current_cycle_summary])

        self.__update_best_models()
        return None

    def __update_best_models(self
                             ) -> None:
        validation_set = self.__epoch_results.loc[(
            slice(None), slice(None), ['test'], slice(None), slice(None)), :]

        if not validation_set.empty:
            logger.debug(f'{validation_set=}')
            for column_name in validation_set.columns:
                column = validation_set[column_name]
                metric = column_name.split('.')[1].lower()
                indices = column.idxmin(skipna=True)
                if not isinstance(indices, tuple):
                    continue
                if self.__lower_is_better[metric]:
                    model_id, graph_name, _, iteration = column.idxmin(skipna=True)
                    measurements = column.min(skipna=True)
                else:
                    model_id, graph_name, _, iteration = column.idxmax(skipna=True)
                    measurements = column.max(skipna=True)

                self.__best_results[column_name] = {'model_id': model_id,
                                                    'graph_name': graph_name,
                                                    'iteration': iteration,
                                                    'measurements': measurements}
        return None

    def __measurements_from_pytorch(self, 
                                    measurements: dict
                                    ) -> dict:
        ret_measurements = {}
        for k, v in measurements.items():
            should_k_be_ignored = any([k.find(ig) >= 0 for ig in self.__ignore_measurements])
            if should_k_be_ignored:
                continue

            if isinstance(v, dict):
                for inner_k, v in self.__measurements_from_pytorch(v).items():
                    ret_measurements[f'{inner_k}.{k}'] = v
            elif isinstance(v, torch.Tensor):
                ret_measurements[k] = v.mean().item()
            else:
                ret_measurements[k] = v
        return ret_measurements

    def __parse_metrics(self
                        ) -> None:
        configs = self.get_config()
        self.__lower_is_better = configs['lower_is_better']
        return None

    def get_config(self
                   ) -> dict:
        return {
            'lower_is_better': defaultdict(lambda: False) | {
                'error': True,
                'loss': True,
                'entropy': True,
            },
            'measurement_name_synonyms': {
                'error': ['error', 'err'],
                'accuracy': ['accuracy', 'acc'],
                'mAP': ['average_precision', 'mean_average_precision', 'mAP', 'AP'],
                'auc': ['area_under_curve', 'AUC']
            },
        }
