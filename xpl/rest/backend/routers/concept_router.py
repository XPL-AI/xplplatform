from typing import Optional, List, Dict
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from xpl.backend.services import concept_service

router = APIRouter()


class ConceptSetup(BaseModel):
    conceptId: Optional[str]
    name: str
    bnId: Optional[str]
    googleImageLabel: Optional[str]
    synsetId: Optional[str]


@router.post("/concept")
async def create_concept(concept_setup: ConceptSetup):
    if concept_setup.conceptId is None:
        concept_setup.conceptId = concept_setup.name

    await concept_service.setup_concept(concept_setup.conceptId, concept_setup.name, concept_setup.bnId)
    return await concept_service.get_concept(concept_setup.conceptId)


@router.get("/concept/{concept_id}")
async def get_concept(concept_id):
    concept = await concept_service.get_concept(concept_id)
    if concept is None:
        raise HTTPException(404, f'Concept concept_id={concept_id} was not found.')

    return concept


@router.get("/concepts/search")
async def search_concept(field, value):
    concepts = await concept_service.search_concept(field, value)
    # if concept is None:
    #     raise HTTPException(404, f'Concept concept_id={concept_id} was not found.')

    return concepts
