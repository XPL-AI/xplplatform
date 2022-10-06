"""
Created Date: Monday, May 3rd 2021, 2:29:56 pm
Author: Ali S. Razavian

Copyright (c) 2021 XPL Technologies AB
"""

import itertools

from xpl.infrastructure.utils import timed
from xpl.dataset.dataset.xpl_dataloader import XPLDataLoader
from xpl.model.neural_net.xpl_model import XPLModel
from xpl.measurer.xpl_measurer import XPLMeasurer

class DAG:
    """
    DAG (Directed Acyclic Graph) is a special type of graph.
    The job of a DAG is to flow information from sources to destinations and visit any intermidiate
    nodes exactly ones.
    Upon executing a dag, batch[heads] will go through the neural networks and the output of the network
    is put back into the batch as batch[tails]
    DAG can have multiple heads and tails
    DAG can have multiple measurers
    DAG only has ``ONE`` data_loader though. 
    """

    def __init__(self,
                 name: str,
                 definition: dict,
                 data_loader: XPLDataLoader,  # DAG only has one data loader
                 models: dict[str, XPLModel],
                 measurers: dict[str, XPLMeasurer],
                 ):
        self.__name = name
        self.__definition = definition
        self.__models = models
        self.__data_loader = data_loader
        self.__measurers = measurers

        self.__dag_structure = {
            model.get_name(): {
                'heads': model.get_heads(),
                'tails': model.get_tails(),
            } for model in self.__models.values()
        }

        self.__sorted_model_names = self.__sort_model_names(self.__dag_structure)

    def get_input_sets(self):
        return self.__data_loader.get_input_sets()

    def __str__(self
                ) -> str:
        model_str = ', '.join(str(model) for model in self.__models.values())
        data_loader_str = str(self.__data_loader)
        measurer_str = ', '.join(str(measurer) for measurer in self.__measurers.values())

        return ',\n\t'.join([f'graph_name:{self.__name}',
                             f'models: {model_str}',
                             f'data_loader: {data_loader_str}',
                             f'measurers: {measurer_str}',
                             '\n'
                             ])

    def __sort_model_names(self,
                           structure: list[tuple]
                           ) -> list[str]:
        set_of_heads = set(itertools.chain(*[edge['heads'] for edge in structure.values()]))
        set_of_tails = set(itertools.chain(*[edge['tails'] for edge in structure.values()]))
        list_of_models = list(structure.keys())

        set_of_sources = set_of_heads - set_of_tails

        sorted_list_of_models = []
        num_edges = len(list_of_models)
        for i in range(num_edges):
            for model_name in list_of_models:
                if not bool(set(structure[model_name]['heads']) - set_of_sources):
                    list_of_models.remove(model_name)
                    sorted_list_of_models.append(model_name)
                    [set_of_sources.add(tail) for tail in structure[model_name]['tails']]
                    break

        return sorted_list_of_models

    @timed
    def execute(self,
                input_set: str,
                ) -> dict:

        is_train = input_set == 'training'
        batch = self.__data_loader.get_next_batch(input_set=input_set)
        if batch is None:
            return None

        for model_name in self.__sorted_model_names:
            self.__models[model_name](batch=batch)

        measurements = {}
        for measurer_name, measurer in self.__measurers.items():
            measurement = measurer(batch=batch, is_train=is_train)
            measurements = self.__merge_measurements(measurements=measurements,
                                                     new_measurement=measurement,
                                                     measurer_name=measurer_name)

        self.__data_loader.record_predictions_and_measurements(input_set=input_set,
                                                               measurements=measurements,
                                                               batch=batch)

        return measurements

    def __merge_measurements(self,
                             measurements: dict,
                             new_measurement: dict,
                             measurer_name: str):
        # Measurement has the following form:
        # {'loss': {'measurement_name (e.g.) english, pedestrian_keypoints, etc): value }
        #  'error': {blah blah},
        #  'blah blah': 'blah blah',
        # }
        for k, v in new_measurement.items():
            if k not in measurements.keys():
                measurements[k] = {}

            measurements[k] |= {measurer_name: v}

        return measurements
