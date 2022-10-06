from typing import Optional, List, Dict
from pydantic import BaseModel


class ConceptDefinition(BaseModel):
    concept_id: Optional[str]
    wn_id: Optional[str]
    # bn_id: Optional[str]
    user_provided_concept_name: str
    user_provided_concept_definition: Optional[str]


class TaskDefinition(BaseModel):
    name: Optional[str]
    modality: Optional[str]
    model_size: Optional[str]
    concept_definitions: Optional[List[ConceptDefinition]]


class TaskConcept(BaseModel):
    display_name: str
    concept_id: str


class ModelComponent(BaseModel):
    url: str
    name: str


class Model(BaseModel):
    model_id: Optional[str]
    components: Dict[str, ModelComponent]
    output: Optional[Dict[str, str]]
    version: int


class Task(BaseModel):
    task_id: str
    name: str
    user_id: str
    client_api_key: str
    modality: str
    concepts: List[TaskConcept]
    model_size: str
    experiment_id: str

    dataset_bucket: Optional[str]

    created_on: Optional[str]
    modified_on: Optional[str]
    model: Optional[Model]


class Modality(BaseModel):
    modality_id: str
    supported_model_sizes: Optional[List[str]]
