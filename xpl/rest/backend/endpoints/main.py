import time

import fastapi
from starlette.responses import JSONResponse

from xpl.rest.backend.routers import \
    configuration_router, sampling_router, \
    customer_router, model_router, task_router, \
    concept_router, annotation_router

from xpl.backend.services import authentication_service


client_api = fastapi.FastAPI()
client_api.include_router(configuration_router.router)
client_api.include_router(sampling_router.router)
client_api.include_router(customer_router.router)
client_api.include_router(model_router.router)
client_api.include_router(task_router.router)
client_api.include_router(concept_router.router)
client_api.include_router(annotation_router.router)

# client_api.include_router(
#     model.router,
#     prefix="/items",
#     tags=["items"],
#     dependencies=[Depends(get_token_header)],
#     responses={404: {"description": "Not found"}},
# )


@client_api.middleware("http")
async def add_process_time_header(request: fastapi.Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response


@client_api.middleware("http")
async def verify_client_key(request: fastapi.Request, call_next):
    if ('ClientKey' not in request.headers) or ('ClientSecret' not in request.headers):
        return JSONResponse(status_code=401)
    secret_is_correct = authentication_service.validate_secret(request.headers['ClientKey'], request.headers['ClientSecret'])
    if not secret_is_correct:
        return JSONResponse(status_code=401)

    response = await call_next(request)

    return response


@client_api.get("/")
async def root():
    return """XPL client API is up!"""
