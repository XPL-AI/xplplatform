import os

from pydantic import BaseModel
from typing import Optional, Dict, List

from fastapi import APIRouter, HTTPException, Request

from xpl.task import TaskService

from xpl.infrastructure.storage import CloudStorageRepository

router = APIRouter()


@router.post("/upload/{task_id}/{data_point_id}")
async def get_upload_url(task_id: str, data_point_id: str, request: Request, file_names: List[str]):
    task_service = TaskService()
    task = task_service.get_task(task_id)
    if ('task_api_key' not in request.headers) or request.headers['task_api_key'] != task.client_api_key:
        raise HTTPException(status_code=401)

    result = dict.fromkeys(file_names)
    repository = CloudStorageRepository(bucket_name=task.dataset_bucket)

    for file_name in file_names:
        blob_name = get_blob_name(task_id=task_id,
                                  data_point_id=data_point_id,
                                  file_name=file_name)
        url_for_upload, gsutil_url = repository.get_url_for_upload(blob_name=blob_name)
        result[file_name] = {'url_for_upload': url_for_upload,
                             'url': gsutil_url}

    return result


def get_blob_name(task_id, data_point_id, file_name):
    blob_name = os.path.join(task_id,
                             str(data_point_id[0]),
                             str(data_point_id[1]),
                             str(data_point_id[2]),
                             str(data_point_id),
                             file_name)
    return blob_name
