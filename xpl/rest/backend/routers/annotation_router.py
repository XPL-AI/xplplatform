from typing import Optional, List, Dict
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from xpl.backend.services import annotations_service
from xpl.backend.services.annotations_service import Annotation, Concept

router = APIRouter()


class LocationModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationModel(BaseModel):
    concept_id: str
    owner_id: str
    sample_file_name: str
    instance_id: str
    source: str
    source_id: str
    timestamp: str
    location: LocationModel


@router.post("/annotation")
async def submit_annotation(annotation_model: AnnotationModel):
    annotation = Annotation(**dict(annotation_model))

    await annotations_service.submit_annotation(annotation)
