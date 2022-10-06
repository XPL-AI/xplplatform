from typing import Optional, List, Dict
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, Request

from xpl.concept import concept_service

from xpl.rest.userapi.routers.authentication import authenticate_request

router = APIRouter()


@router.get("/concept")
async def search_by_lemma(request: Request, lemma: str):
    user_key_info = authenticate_request(request)
    search_results = concept_service.search(user_input_text=lemma)
    return search_results
