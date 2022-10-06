from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from xpl.task import ModalityService, ModalityNotExist

router = APIRouter()


@router.get("/modalities")
async def list_modalities(request: Request):
    modality_service = ModalityService()
    return modality_service.list_modalities()


@router.get("/modality/{modality_id}")
async def get_modality_info(request: Request, modality_id: str):
    modality_service = ModalityService()
    try:
        return modality_service.get_modality(modality_id)
    except ModalityNotExist:
        return JSONResponse(status_code=404)

