from pydantic import BaseModel
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from xpl.infrastructure.storage.repository import CloudStorageRepository

from xpl.task.entities import TaskDefinition
from xpl.task import TaskService, Task, TaskNotFoundException, SetupTaskInvalidInputException

from xpl.rest.userapi.routers.authentication import authenticate_request

router = APIRouter()


class TaskConceptResource(BaseModel):
    display_name: str
    concept_id: str


class ModelComponentResource(BaseModel):
    url_for_download: str
    name: str


class ModelResource(BaseModel):
    model_id: Optional[str]
    components: Dict[str, ModelComponentResource]
    mobile_components: Optional[Dict[str, ModelComponentResource]]
    output: Optional[Dict[str, str]]
    version: int


class TaskResource(BaseModel):
    task_id: str
    name: str
    client_api_key: str
    modality: str
    concepts: List[TaskConceptResource]
    model_size: str
    created_on: Optional[str]
    model: Optional[ModelResource]


class PublicTaskResource(BaseModel):
    task_id: str
    name: str
    concepts: List[TaskConceptResource]
    model: Optional[ModelResource]


@router.post("/task")
async def setup_task_from_definition(request: Request, task_definition: TaskDefinition):
    user_key_info = authenticate_request(request)

    task_service = TaskService()
    try:
        created_task = task_service.setup_task_from_definition(user_id=user_key_info['user_id'],
                                                               task_definition=task_definition)

    except SetupTaskInvalidInputException as e:
        return JSONResponse(__build_errors_response(e.args), status_code=422)

    return __to_private_resource(created_task)


@router.get("/task/{task_id}")
async def get_task_by_id(request: Request, task_id: str):
    if 'api_key' in request.headers:
        user_key_info = authenticate_request(request)
        user_id = user_key_info['user_id']

        task_service = TaskService()

        try:
            task = task_service.get_task(task_id)

            if task.user_id != user_id:
                raise HTTPException(status_code=404)

            return __to_private_resource(task)
        except TaskNotFoundException:
            raise HTTPException(status_code=404)
    elif 'task_api_key' in request.headers:
        try:
            task_service = TaskService()
            task = task_service.get_task(task_id)
            if task.client_api_key != request.headers['task_api_key']:
                raise HTTPException(status_code=404)
            return __to_public_resource(task=task)
        except TaskNotFoundException:
            raise HTTPException(status_code=404)


@router.get("/tasks")
async def list_user_tasks(request: Request):
    user_key_info = authenticate_request(request)
    user_id = user_key_info['user_id']

    task_service = TaskService()

    tasks = task_service.list_user_tasks(user_id=user_id)

    return [__to_private_resource(task) for task in tasks]


def __to_private_resource(task: Task):
    task_resource = TaskResource(task_id=task.task_id,
                                 name=task.name,
                                 client_api_key=task.client_api_key,
                                 modality=task.modality,
                                 model_size=task.model_size,
                                 concepts=[],
                                 created_on=task.created_on)

    for concept in task.concepts:
        task_resource.concepts.append(TaskConceptResource(concept_id=concept.concept_id,
                                                          display_name=concept.display_name))

    if task.model:
        task_resource.model = ModelResource(model_id=task.model.model_id,
                                            version=task.model.version,
                                            components={},
                                            output={})
        for key, component in task.model.components.items():
            url_for_download = CloudStorageRepository().get_url_for_download_by_uri(component.url)
            task_resource.model.components[key] = ModelComponentResource(url_for_download=url_for_download,
                                                                         name=component.name)

        for key, concept_id in task.model.output.items():
            task_resource.model.output[key] = concept_id

    return task_resource


def __to_public_resource(task: Task):
    task_resource = PublicTaskResource(task_id=task.task_id,
                                       name=task.name,
                                       concepts=[])

    for concept in task.concepts:
        task_resource.concepts.append(TaskConceptResource(concept_id=concept.concept_id,
                                                          display_name=concept.display_name))

    if task.model:
        task_resource.model = ModelResource(model_id=task.model.model_id,
                                            version=task.model.version,
                                            components={},
                                            output={})
        for key, component in task.model.components.items():
            url_for_download = CloudStorageRepository().get_url_for_download_by_uri(component.url,
                                                                                    time_to_live_seconds=3600)
            task_resource.model.components[key] = ModelComponentResource(url_for_download=url_for_download,
                                                                         name=component.name)

        if task.model.mobile_components is not None:
            task_resource.model.mobile_components = {}
            for key, component in task.model.mobile_components.items():
                url_for_download = CloudStorageRepository().get_url_for_download_by_uri(component.url,
                                                                                        3600)
                task_resource.model.mobile_components[key] = ModelComponentResource(url_for_download=url_for_download,
                                                                                    name=component.name)

        for key, concept_id in task.model.output.items():
            task_resource.model.output[key] = concept_id

    return task_resource


def __build_errors_response(exception_args):
    errors = []
    for arg in exception_args:
        errors.append(str(arg))
    return {'errors': errors}
