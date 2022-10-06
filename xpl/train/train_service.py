import logging

from typing import List

from xpl.task.entities import TaskConcept
from xpl.train.dag import DAG

logger = logging.getLogger(__name__)


class TrainService:

    def __init__(self
                 ) -> None:
        logger.info('___Train Service initiated___')

    def get_dags(self,
                 definitions: dict,
                 models: dict,
                 data_loaders: dict,
                 measurers: dict):
        dags = {}
        for name, definition in definitions.items():
            dags[name] = self.get_dag(name=name,
                                      definition=definition,
                                      models=models,
                                      data_loaders=data_loaders,
                                      measurers=measurers,
                                      )
        return dags

    def get_dag(self,
                name: str,
                definition: dict,
                models: dict,
                data_loaders: dict,
                measurers: dict,
                ) -> DAG:

        dag_data_loader = data_loaders[definition['data_loader']]

        dag_models = {
            model_name: models[model_name] for model_name in definition['models']
        }
        dag_measurers = {
            measurer_name: measurers[measurer_name] for measurer_name in definition['measurers']
        }

        return DAG(name=name,
                   definition=definition,
                   data_loader=dag_data_loader,
                   models=dag_models,
                   measurers=dag_measurers)

    def get_experiment_definition(self,
                                  task_name: str,
                                  task_type: str,
                                  model_size: str,
                                  concepts: List[TaskConcept],
                                  modality: str,
                                  ) -> dict:
        concept_groups = self.__get_concept_groups(concepts=concepts,
                                                   task_name=task_name)
        # TODO this is the function that defines the neural networks, Models, DAGs and everything else.
        # This is the magical being that for every customers generate a pipeline that suits their needs
        print(f'{task_type=}, {modality=} {model_size=}')
        if task_type == 'recognition':
            return {
                'models': {
                    'image_rep': {
                        'heads': [modality],
                        'tails': [f'{modality}_representation'],
                        'input_size': self.__get_input_size(model_size=model_size,
                                                            modality=modality),
                        'type': 'backbone',
                        'modality': modality,
                        'model_size': model_size,
                    },
                } | {
                    f'{concept_group_name}_head': {
                        'heads': [f'{modality}_representation'],
                        'tails': [f'{concept_group_name}'],
                        'type': f'{modality}_recognition',
                        'input_channels': self.__get_head_size(model_size=model_size,
                                                               modality=modality),
                        'modality': modality,
                        'model_size': model_size,
                        'output_channels': len(concept_ids),  # +1 for background
                        'output_channel_names': concept_ids
                    } for concept_group_name, concept_ids in concept_groups.items()
                },
                'measurers': {
                    f'{concept_group_name}': {
                        'type': f'{modality}_recognition',
                        'concept_group_name': [f'{concept_group_name}'],
                        'num_classes': len(concept_ids),  # +1 for everything else
                        'output_is_probability': False,
                    } for concept_group_name, concept_ids in concept_groups.items()
                },
                'data_loaders': {
                    f'{concept_group_name}_data_loader': {
                        'input_size': self.__get_input_size(model_size=model_size,
                                                            modality=modality),
                        'output_size': self.__get_head_size(model_size=model_size,
                                                            modality=modality),

                        'input_name': modality,
                        'transform': 'random',
                        'has_targets': True,
                        'targets': {
                            concept_group_name: concept_ids,
                        }
                    } for concept_group_name, concept_ids in concept_groups.items()
                },
                'dags': {
                    concept_group_name: {
                        'models': ['image_rep',  f'{concept_group_name}_head'],
                        'measurers': [f'{concept_group_name}'],
                        'data_loader': f'{concept_group_name}_data_loader',
                    } for concept_group_name, concept_ids in concept_groups.items()
                }
            }

    def __get_input_size(self,
                         model_size: str,
                         modality: str
                         ) -> dict:
        if modality == 'image':
            if model_size == 'onesize':
                return {
                    'input_width': 512,
                    'input_height': 320,
                    'input_channels': 3,
                }
        if modality == 'audio':
            if model_size == 'onesize':
                return {
                    'input_width': 10 * 16000,  # 10 seconds
                    'input_channels': 1,
                }
    # This number comes from the backbone. We need to find a good place for it!

    def __get_head_size(self,
                        model_size: str,
                        modality: str
                        ) -> dict:
        if modality == 'image':
            if model_size == 'onesize':
                return {
                    'target_width': 16,
                    'target_height': 10,
                    'target_channels': 1280,
                }

        if modality == 'audio':
            if model_size == 'onesize':
                return {
                    'target_channels': 512,
                }

    def __get_concept_groups(self,
                             concepts: list[TaskConcept],
                             task_name: str):
        # 'xpl:wn:n5933834' is background. 
        # TODO: We need to come up with a better name for background
        return {
            task_name: ['xpl:wn:n5933834'] + [concept.concept_id for concept in concepts]
        }
