import fastapi

from xpl.rest.dataapi.routers import datapoint, storage_event_handler, upload

api = fastapi.FastAPI()
api.include_router(datapoint.router)
# data_api.include_router(storage_event_handler.router)
api.include_router(upload.router)
