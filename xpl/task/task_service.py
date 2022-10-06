import uuid

from datetime import datetime
from typing import List

from google.cloud import firestore

from xpl.concept import concept_service
from xpl.concept.concept_service import Concept

from xpl.data import DataService

from xpl.task import config
from xpl.task.entities import TaskDefinition, Task, TaskConcept, Model, ModelComponent

from xpl.user import UserService

TASK_COLLECTION_NAME = 'tasks'
FIRESTORE_CLIENT = firestore.Client()


class TaskService:
    def __init__(self):
        global FIRESTORE_CLIENT
        if FIRESTORE_CLIENT is None:
            FIRESTORE_CLIENT = firestore.Client()
        self.__firestore_client = FIRESTORE_CLIENT

    def setup_task_from_definition(self,
                                   user_id: str,
                                   task_definition: TaskDefinition
                                   ) -> Task:

        self.__validate_task_definition(task_definition)

        errors = []
        task_concepts: List[TaskConcept] = []
        concepts_to_create = []

        for concept_definition in task_definition.concept_definitions:
            if concept_definition.concept_id is not None:
                """Concept was registered at XPL before and user selected it for this task definition"""
                concept = concept_service.get_by_id(concept_definition.concept_id)
                if concept is None:
                    errors.append(f'Concept concept_id={concept_definition.concept_id} does not exist.')
                    continue
            else:
                """Concept was not registered at XPL before."""
                if concept_service.CONCEPT_SEARCH_PROVIDER == 'wordnet' \
                        and concept_definition.wn_id is not None:
                    """Concept definition was acknowledged with wordnet's concept."""
                    wn_id = concept_definition.wn_id.lower()
                    concept = concept_service.get_by_external_id(external_id=wn_id)
                    if concept is None:
                        errors.append(f'Concept with wn_id="{wn_id}" does not exist."')
                        continue
                    if concept.concept_id is None:
                        concept.concept_id = f'xpl:{wn_id}'
                        concepts_to_create.append(concept)
                else:
                    """Concept was not acknowledged with any external conception and will be created from user's input."""
                    if not concept_definition.user_provided_concept_name:
                        errors.append(f'Concept definition is empty.')
                    concept = Concept(
                        concept_id=f'xpl:user:{str(uuid.uuid4())[-12:]}',
                        name=concept_definition.user_provided_concept_name,
                        definition=concept_definition.user_provided_concept_definition)

                    concepts_to_create.append(concept)

            task_concepts.append(TaskConcept(
                display_name=concept_definition.user_provided_concept_name,
                concept_id=concept.concept_id
            ))

        if len(errors) > 0:
            raise SetupTaskInvalidInputException(errors)

        """Create missing concepts"""
        for concept in concepts_to_create:
            concept_service.create(concept)

        """Setup experiment"""
        task_id = self.__new_task_id()
        self.__setup_experiment(user_id=user_id,
                                task_id=task_id,
                                modality=task_definition.modality)

        user = UserService().get_user_by_id(user_id)

        """Save task"""
        task_client_api_key = uuid.uuid4().hex
        task: Task = Task(
            task_id=task_id,
            name='Not set' if task_definition.name is None else task_definition.name,
            user_id=user_id,
            client_api_key=task_client_api_key,
            modality=task_definition.modality,
            model_size=task_definition.model_size,
            # model_id=]
            experiment_id=task_id,

            dataset_bucket=user.datasets_storage_bucket,
            concepts=task_concepts,
            created_on=str(datetime.utcnow())
        )

        self.__firestore_client \
            .collection(TASK_COLLECTION_NAME) \
            .document(task_id) \
            .set(task.dict())

        return task

    def update_task_active_model(self,
                                 task_id: str,
                                 model: Model
                                 ) -> Task:
        task = self.get_task(task_id=task_id)

        task.model = model

        self.__firestore_client \
            .collection(TASK_COLLECTION_NAME) \
            .document(task_id) \
            .set(task.dict())

        return task

    def get_task(self, task_id
                 ) -> Task:
        task_doc = self.__firestore_client.collection(TASK_COLLECTION_NAME).document(task_id).get()
        if not task_doc.exists:
            raise TaskNotFoundException(f'task_id{task_id} not found in collection={TASK_COLLECTION_NAME}')

        return Task(**task_doc.to_dict())

    def verify_task_api_key(self,
                            task_id: str,
                            client_api_key: str):
        task_doc = self.__firestore_client.collection(TASK_COLLECTION_NAME).document(task_id).get()
        if not task_doc.exists:
            return False
        if task_doc.to_dict()['client_api_key'] == client_api_key:
            return True
        return False

    def list_user_tasks(self, user_id
                        ) -> List[Task]:
        collection_ref = self.__firestore_client.collection(TASK_COLLECTION_NAME)
        query_ref = collection_ref.where('user_id', u'==', user_id)

        documents = query_ref.stream()
        result: List[Task] = []
        for doc in documents:
            result.append(Task(**doc.to_dict()))

        return result

    def delete_task(self, task_id):
        raise NotImplemented

    def __new_task_id(self):
        return uuid.uuid4().hex

    def __setup_experiment(self,
                           user_id,
                           task_id,
                           modality
                           ):
        data_service = DataService()
        data_service.setup_dataset(user_id=user_id,
                                   task_id=task_id,
                                   modality=modality)

    def __validate_task_definition(self, task_definition: TaskDefinition):
        errors = []
        if task_definition is None:
            errors.append(f'Task definition empty.')
            raise SetupTaskInvalidInputException(errors)

        supported_modalities = config['supported_modalities']
        planned_supported_modalities_text = config['planned_supported_modalities_text']

        if task_definition.modality is None:
            errors.append(f'Task definition misses required argument: modality.\n'
                          f'Supported modalities: {supported_modalities}\n'
                          f'{planned_supported_modalities_text}')
        elif task_definition.modality not in supported_modalities:
            errors.append(f'Modality: "{task_definition.modality}" is not a supported modality.\n'
                          f'Supported modalities: {supported_modalities}\n'
                          f'{planned_supported_modalities_text}')

        if task_definition.model_size is None:
            error = f'Task definition misses required argument: model_size.'
            if task_definition.modality is not None \
                    and task_definition.modality in supported_modalities:
                supported_model_sizes_for_modality = config["supported_model_sizes"][task_definition.modality]
                error += f'Modality="{task_definition.modality}" supports model sizes: {supported_model_sizes_for_modality}'
            errors.append(error)
        elif task_definition.modality is not None \
                and task_definition.modality in supported_modalities:
            supported_model_sizes_for_modality = config["supported_model_sizes"][task_definition.modality]
            if task_definition.model_size not in supported_model_sizes_for_modality:
                error = f'model_size "{task_definition.model_size}" is not available for modality "{task_definition.modality}".'
                error += f'Modality="{task_definition.modality}" supports model sizes: {supported_model_sizes_for_modality}'
                errors.append(error)

        if task_definition.concept_definitions is None:
            errors.append(f'Task definition misses required argument: concept_definitions.')
        elif len(task_definition.concept_definitions) < 1:
            errors.append(f'Task definition should contain at least 1 concept_definition.')

        if len(errors) > 0:
            raise SetupTaskInvalidInputException(*errors)


class SetupTaskInvalidInputException(Exception):
    pass


class TaskNotFoundException(Exception):
    pass
