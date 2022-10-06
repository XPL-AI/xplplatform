from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, Request, Response

from xpl.task import TaskService

from xpl.data import DataService
from xpl.data.repositories import DataItem

from xpl.rest.dataapi.dto import DataPoint


router = APIRouter()


@router.post("/datapoint")
async def create_data_point(request: Request, data_point: DataPoint):
    # TODO: a call to TaskService has to be cached.
    task = TaskService().get_task(task_id=data_point.task_id)
    if ('task_api_key' not in request.headers) or request.headers['task_api_key'] != task.client_api_key:
        raise HTTPException(status_code=401)
    data_items = []

    for item in data_point.data_items:
        data_item = DataItem(data_point_id=data_point.data_point_id,
                             data_point_file=';'.join(data_point.file_uris),
                             timestamp=datetime.utcnow(),
                             concept_id=item.concept_id,
                             instance_id=item.instance_id,
                             predictor_type=item.predictor_type,
                             predictor_id=item.predictor_id,
                             input_set=item.input_set,
                             log_informativeness=item.log_informativeness,
                             logit_confidence=item.logit_confidence,
                             location=item.location,
                             text=item.text,
                             value=item.value,
                             previous_data_point_id=data_point.previous_data_point_id,
                             next_data_point_id=data_point.next_data_point_id,
                             collected_by_device_fingerprint_id=data_point.collected_by_device_fingerprint_id)
        data_items.append(data_item)

    data_service = DataService()
    data_service.ingest_data_items(user_id=task.user_id,
                                   task_id=task.task_id,
                                   processed=False,
                                   data_items=data_items)

    return Response(status_code=200)


@router.post("/datapoints")
async def create_data_points(request: Request, data_points: List[DataPoint]):
    unique_task_ids = set(map(lambda x: x.task_id, data_points))
    if len(unique_task_ids) > 1:
        return Response(status_code=422, content='DataPoints should belong to the same task')

    # TODO: a call to TaskService has to be cached.
    task = TaskService().get_task(task_id=data_points[0].task_id)
    if ('task_api_key' not in request.headers) or request.headers['task_api_key'] != task.client_api_key:
        raise HTTPException(status_code=401, detail='Unauthorized')

    data_items = []
    for data_point in data_points:
        for item in data_point.data_items:
            data_item = DataItem(data_point_id=data_point.data_point_id,
                                 data_point_file=';'.join(data_point.file_uris),
                                 timestamp=datetime.utcnow(),
                                 concept_id=item.concept_id,
                                 instance_id=item.instance_id,
                                 predictor_type=item.predictor_type,
                                 predictor_id=item.predictor_id,
                                 input_set=item.input_set,
                                 log_informativeness=item.log_informativeness,
                                 logit_confidence=item.logit_confidence,
                                 location=item.location,
                                 text=item.text,
                                 value=item.value,
                                 previous_data_point_id=data_point.previous_data_point_id,
                                 next_data_point_id=data_point.next_data_point_id,
                                 collected_by_device_fingerprint_id=data_point.collected_by_device_fingerprint_id)
            data_items.append(data_item)

    data_service = DataService()
    data_service.ingest_data_items(user_id=task.user_id,
                                   task_id=task.task_id,
                                   processed=False,
                                   data_items=data_items)

    return Response(status_code=200)
