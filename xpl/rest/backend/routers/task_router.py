import json
from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from xpl.backend.services import task_service

router = APIRouter()


class ConceptMd(BaseModel):
    conceptId: Optional[str]
    name: Optional[str]
    bnId: Optional[str]


class ModelSetupMd(BaseModel):
    name: str
    modelFileUri: str
    outputs: Dict[str, ConceptMd]


class TaskSetupMd(BaseModel):
    customerId: str
    size: Optional[str]
    modality: str
    name: str
    concepts: Optional[List[ConceptMd]]
    modelSetup: Optional[ModelSetupMd]


@router.post("/task")
async def create_task(task_setup: TaskSetupMd):
    """Setup task and generate model"""
    if task_setup.concepts is not None:
        concepts = []
        for c in task_setup.concepts:
            if c.conceptId is None or c.conceptId == '':
                concept_id = c.name.replace(' ', '_').lower()
                c.conceptId = concept_id

            concepts.append(dict(c))

        task = await task_service.setup_new_task(task_setup.customerId, task_setup.name,
                                                 task_setup.modality, concepts,
                                                 task_setup.size)
        return task

    """Setup task from existing model"""
    if task_setup.modelSetup is not None:
        concepts = []
        output = {}
        for out_index, concept in task_setup.modelSetup.outputs.items():
            if concept.conceptId is None or concept.conceptId == '':
                concept.conceptId = concept.name.replace(' ', '_').lower()
            concepts.append(dict(concept))

            output[str(out_index)] = {'name': concept.name, 'conceptId': concept.conceptId}

        task = await task_service.setup_from_model(task_setup.customerId, task_setup.name,
                                                   task_setup.modality, concepts,
                                                   task_setup.modelSetup.name, task_setup.modelSetup.modelFileUri,
                                                   output, task_setup.size)

        return task

    raise HTTPException(400, 'Task should contain either list of concepts, or modelSetup. Both were None.')


@router.get("/task/{task_id}/model")
async def get_model(task_id: str):
    model = await task_service.get_model(task_id)
    del model['modelFileUri']

    return json.dumps(model)

