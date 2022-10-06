from datetime import datetime
import os
from typing import List

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from xpl.infrastructure.storage.repository import CloudStorageRepository

from xpl.annotation import AnnotationService, SimpleAnnotationJob, SimpleInstanceFrame


router = APIRouter()

templates_dir = os.path.join(os.environ['XPL_CODE_DIR'], 'xpl', 'rest', 'annotationapi', 'templates')
templates = Jinja2Templates(directory=templates_dir)


@router.get("/jobs/{task_id}")
async def list_annotation_jobs(request: Request, task_id: str):
    annotation_service = AnnotationService()
    return annotation_service.list_annotation_jobs(task_id=task_id)


@router.get("/job/{annotation_job_id}")
async def get_annotation_job(request: Request, annotation_job_id: str):
    annotation_service = AnnotationService()
    annotation_job = annotation_service.get_annotation_job(annotation_job_id=annotation_job_id)

    for key_row_id, instance_frame in annotation_job.instance_frames.items():
        url_for_download = CloudStorageRepository().get_url_for_download_by_uri(instance_frame.data_point_file,
                                                                                time_to_live_seconds=28800)
        instance_frame.data_point_file = url_for_download

    return annotation_job


@router.post("/job/{annotation_job_id}")
async def submit_annotation_job(request: Request, simple_annotation_job: SimpleAnnotationJob):
    annotation_service = AnnotationService()
    annotation_service.submit_annotation_job(simple_annotation_job=simple_annotation_job)


@router.get("/job/{annotation_job_id}/annotate", response_class=HTMLResponse)
async def annotate_ui(request: Request, annotation_job_id: str):
    return templates.TemplateResponse("tagger.html", {"request": request, "annotation_job_id": annotation_job_id})


@router.get("/", response_class=HTMLResponse)
async def index_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/audio_annotation", response_class=HTMLResponse)
async def index_ui(request: Request):
    return templates.TemplateResponse("audio_annotator.html", {"request": request})