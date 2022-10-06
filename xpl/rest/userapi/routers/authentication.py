from typing import Optional, List, Dict
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse

from xpl.user import UserService, InvalidCredentials

router = APIRouter()


class Credentials(BaseModel):
    username: str
    secret: str


@router.post("/login")
async def login(credentials: Credentials):
    user_service = UserService()
    try:
        api_key = credentials.secret
        verification_result, key = user_service.verify_api_key(username=credentials.username,
                                                               api_key=api_key)
        if verification_result is True:
            return {'api_key': key['key']}
    except Exception as e:
        return JSONResponse(status_code=401)

    return JSONResponse(status_code=401)


def authenticate_request(request: Request):
    user_service = UserService()

    if ('username' not in request.headers) or ('api_key' not in request.headers):
        raise HTTPException(status_code=401)

    try:
        verification_result, key = user_service.verify_api_key(username=request.headers['username'],
                                                               api_key=request.headers['api_key'])

        if verification_result is not True:
            raise HTTPException(status_code=401)
        else:
            return key
    except InvalidCredentials:
        raise HTTPException(status_code=401)

