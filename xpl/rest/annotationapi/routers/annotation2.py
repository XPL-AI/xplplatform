import os

from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates

from starlette.responses import JSONResponse

from xpl.user import UserService, InvalidCredentials

from xpl.annotation import AnnotationService2, SimpleAnnotationJob, SimpleInstanceFrame, AnnotationDataPoint
from xpl.infrastructure.storage import CloudStorageRepository

router = APIRouter()

templates_dir = os.path.join(os.environ['XPL_CODE_DIR'], 'xpl', 'rest', 'annotationapi', 'templates')
templates = Jinja2Templates(directory=templates_dir)


@router.get("/task/task_id/annotation-jobs")
async def list_annotation_jobs(request: Request, task_id: str):
    annotation_service = AnnotationService2()
    return annotation_service.list_annotation_jobs_for_task(task_id=task_id)


@router.get("/user/annotation-jobs")
async def list_annotation_jobs(request: Request):
    user_key_info = authenticate_request(request)
    user_id = user_key_info['user_id']

    annotation_service = AnnotationService2()
    return annotation_service.list_annotation_jobs_for_user(user_id=user_id)


@router.get("/annotation-job/{annotation_job_id}/info")
async def get_annotation_job(request: Request, annotation_job_id: str):
    authenticate_request(request)

    annotation_service = AnnotationService2()
    return annotation_service.get_annotation_job_by_id(annotation_job_id=annotation_job_id)


@router.get("/annotation-job/{annotation_job_id}")
async def load_annotation_job(request: Request, annotation_job_id: str):
    authenticate_request(request)

    annotation_service = AnnotationService2()
    job = annotation_service.load_annotation_job(annotation_job_id=annotation_job_id)

    for key_row_id, data_point in job.data_points.items():
        url_for_download = CloudStorageRepository().get_url_for_download_by_uri(data_point.data_point_file,
                                                                                time_to_live_seconds=28800)
        data_point.data_point_file = url_for_download

    job.annotation_job_bucket_name = None
    job.annotation_job_file_name = None

    return job


@router.put("/annotation-job/{annotation_job_id}/datapoint/{data_point_id}")
async def save_annotation_data_point(request: Request, annotation_job_id: str, annotation_data_point: AnnotationDataPoint):
    annotation_service = AnnotationService2()
    await annotation_service.save_data_point(data_point=annotation_data_point, annotation_job_id=annotation_job_id)


@router.post("/annotation-job/{annotation_job_id}")
async def submit_annotation_job(request: Request, annotation_job_id):
    annotation_service = AnnotationService2()
    annotation_service.submit_annotation_job(annotation_job_id=annotation_job_id)


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