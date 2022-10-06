import fastapi
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import firestore

from xpl.rest.userapi.routers import authentication, concept, modality, task

api = fastapi.FastAPI()
api.include_router(authentication.router)
api.include_router(concept.router)
api.include_router(modality.router)
api.include_router(task.router)

CORS_COLLECTION_NAME = 'cors'
FIRESTORE_CLIENT = firestore.Client()


def load_origins():
    global FIRESTORE_CLIENT
    if FIRESTORE_CLIENT is None:
        FIRESTORE_CLIENT = firestore.Client()
    cors_doc = FIRESTORE_CLIENT.collection(CORS_COLLECTION_NAME).document('userapi').get()
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

# @user_api.middleware("http")
# async def add_print_body_header(request: fastapi.Request, call_next):
#     print(await request.body())
#     response = await call_next(request)
#
#     return response
