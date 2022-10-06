import base64
import json

from datetime import datetime

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from xpl.data import DataService
from xpl.data.repositories import DataItem

from xpl.task import TaskService

from xpl.infrastructure.storage.repository_async import download_to_memory, make_uri

from xpl.rest.dataapi.dto import DataPoint

router = APIRouter()


@router.post("/storage/created")
async def handle_storage_created_event(request: Request):
    try:
        # data = await validate_and_parse_request(request)
        data = await validate_and_parse_eventarc_request(request)

        if data["name"].endswith('data_point.json'):
            uri = make_uri(bucket_name=data["bucket"],
                           resource_name=data["name"])

            download_result = await download_to_memory(gsutil_uri=uri)
            if download_result.status == '200':
                data = json.loads(download_result.downloaded_bytes.decode("UTF-8"))
                data_point = DataPoint(**data)
                data_items = []
                for item in data_point.data_items:
                    data_item = DataItem(data_point_id=data_point.data_point_id,
                                         data_point_file=';'.join(data_point.file_uris),
                                         timestamp=data['timeCreated'] if 'timeCreated' in data else datetime.utcnow(),
                                         concept_id=item.concept_id,
                                         instance_id=item.instance_id,
                                         predictor_type=item.predictor_type,
                                         predictor_id=item.predictor_id,
                                         input_set=item.input_set,
                                         log_informativeness=item.log_informativeness,
                                         logit_confidence=item.logit_confidence,
                                         location=item.location,
                                         previous_data_point_id=data_point.previous_data_point_id,
                                         next_data_point_id=data_point.next_data_point_id,
                                         collected_by_device_fingerprint_id=data_point.collected_by_device_fingerprint_id)
                    data_items.append(data_item)

                # TODO: a call to TaskService has to be cached.
                task = TaskService().get_task(task_id=data_point.task_id)
                data_service = DataService()

                data_service.ingest_data_items(user_id=task.user_id,
                                               task_id=data_point.task_id,
                                               processed=False,
                                               data_items=data_items)
                return Response(status_code=204)

        return Response(status_code=204)

    except BadRequestException as e:
        return PlainTextResponse(status_code=400, content=f"Bad Request: {e.args[0]}")

    except Exception as e:
        # log
        print(f"error: {e}")
        return PlainTextResponse(status_code=400, content=f"Bad Request: {e.args[0]}")


async def validate_and_parse_request(request: Request
                                     ) -> dict:
    envelope = await request.json()
    print(await request.body())
    if not envelope:
        msg = "no Pub/Sub message received"
        print(f"error: {msg}")
        raise BadRequestException(msg)

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = f"invalid Pub/Sub message format\n\n{request.body()}"
        print(f"error: {msg}")
        raise BadRequestException(msg)

    # Decode the Pub/Sub message.
    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        try:
            data = json.loads(base64.b64decode(pubsub_message["data"]).decode())

        except Exception as e:
            msg = (
                "Invalid Pub/Sub message: "
                "data property is not valid base64 encoded JSON"
            )
            print(f"error: {e}")
            raise BadRequestException(msg)

        # Validate the message is a Cloud Storage event.
        if not data["name"] or not data["bucket"]:
            msg = (
                "Invalid Cloud Storage notification: "
                "expected name and bucket properties"
            )
            print(f"error: {msg}")
            raise BadRequestException(msg)
        return data


async def validate_and_parse_eventarc_request(request: Request
                                              ) -> dict:
    envelope = await request.json()
    print(await request.body())
    if not envelope:
        msg = "no eventarc message received"
        print(f"error: {msg}")
        raise BadRequestException(msg)

    if not isinstance(envelope, dict) or "protoPayload" not in envelope:
        msg = f"invalid eventarc message format"
        print(f"error: {msg}")
        raise BadRequestException(msg)

    proto_payload = envelope["protoPayload"]

    if isinstance(proto_payload, dict) and "resourceName" in proto_payload:
        try:
            resource_name: str = proto_payload['resourceName']
            split = resource_name.split('/')
            bucket = split[3]
            name = resource_name.split(f'{bucket}/objects/')[-1]
            time_created = envelope['timestamp'] if 'timestamp' in envelope else None
            return {
                'bucket': bucket,
                'name': name,
                'timeCreated': time_created
            }
        except Exception as e:
            msg = (
                "Invalid eventarc message: "
                "data property is not valid base64 encoded JSON"
            )
            print(f"error: {e}")
            raise BadRequestException(msg)


class BadRequestException(Exception):
    pass
