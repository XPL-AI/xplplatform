import json
from typing import List

from fastapi import APIRouter, Request, File, UploadFile

from xpl.backend.services import sampling_service

router = APIRouter()


@router.get("/{model_id}/sample/{sample_id}")
async def get_sample_url(model_id, sample_id):
    return await sampling_service.get_sample(model_id, sample_id)


@router.post("/{model_id}/sample")
async def upload_sample(model_id, request: Request, files: List[UploadFile] = File(...)):
    customer_id = request.headers['ClientKey']

    samples = []
    for file in files:
        data = await file.read()
        samples.append((data, file.filename))

    upload_result = await sampling_service.upload_sample(samples, model_id, customer_id)
    result = json.dumps(upload_result)
    return result


@router.post("/{model_id}/sample_upload_url")
async def upload_sample_over_url(model_id, request: Request, request_body: dict):
    customer_id = request.headers['ClientKey']

    upload_result = await sampling_service.upload_sample_over_upload_url(request_body['sample_sizes'], model_id, customer_id)

    return json.dumps(upload_result)


@router.post("/{model_id}/predictions")
async def upload_predictions(predictions):

    pass


# @router.post("/samples")
# async def upload_samples(files: List[bytes] = File(...)):
#     print(f"file_sizes: {[len(file) for file in files]}")
#     return {"file_sizes": [len(file) for file in files]}

# @router.post("/samples")
# async def upload_samples(request: Request):
#     # print((await request.body()))
#     print(request.headers)
#     print((await request.body()).decode("utf-8", "replace"))
