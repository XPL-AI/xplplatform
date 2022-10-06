import fastapi
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import firestore

from xpl.rest.annotationapi.routers import annotation2


api = fastapi.FastAPI()
api.include_router(annotation2.router)

CORS_COLLECTION_NAME = 'cors'
FIRESTORE_CLIENT = firestore.Client()


def load_origins():
    global FIRESTORE_CLIENT
    if FIRESTORE_CLIENT is None:
        FIRESTORE_CLIENT = firestore.Client()
    cors_doc = FIRESTORE_CLIENT.collection(CORS_COLLECTION_NAME).document('annotationapi').get()
    if not cors_doc.exists:
        return ['http://localhost:3000']

    return cors_doc.to_dict()['allow_origins']


origins = load_origins()

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @annotation_api.middleware("http")
# async def add_process_time_header(request: fastapi.Request, call_next):
#     response = await call_next(request)
#     response.headers["Access-Control-Allow-Origin"] = '*'
#
#     return response
