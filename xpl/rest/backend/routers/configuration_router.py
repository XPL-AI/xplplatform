from fastapi import APIRouter, Request

from xpl.backend.services import configuration_service

router = APIRouter()


@router.get("/configuration")
async def get_configuration(request: Request):
    client_key = request.headers['ClientKey']

    config = configuration_service.get_client_config(client_key)

    content = f'{config}'

    return content
