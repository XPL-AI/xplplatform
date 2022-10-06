import base64
import json
import os
import pytest

import requests

from typing import List

from xpl.rest.tests import config
from xpl.rest.tests.dataapi import fixtures

from xpl.infrastructure.storage import repository_async
from xpl.infrastructure.storage.repository_async import DownloadResult

USER_API_BASE_URI = config["userapi_uri"]
USER_API_AUTH_HEADERS = {'username': config["valid_username"], 'api_key': config["valid_api_key"]}

DATA_API_BASE_URI = config["dataapi_uri"]
APPLICATION_JSON_HEADERS = {'content-type': 'application/json'}

test_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx' \
             b'\xdac\xfc\xff\x9f\xa1\x1e\x00\x07\x82\x02\x7f=\xc8H\xef\x00\x00\x00\x00IEND\xaeB`\x82'


@pytest.mark.asyncio
async def test__upload_samples():
    """Step: Create new task"""
    data = f"""
    {{
        "name": "test_task",
        "modality": "video",
        "model_size": "xxxs",
        "concept_definitions": [
            {{ "user_provided_concept_name": "ONE" }},
            {{ "user_provided_concept_name": "TWO", "concept_id": "xpl:wn:n13743269" }},
            {{ "user_provided_concept_name": "THREE", "wn_id": "wn:n4480415" }}
        ]
    }}
    """

    response = requests.post(url=f'{USER_API_BASE_URI}/task',
                             data=data,
                             headers=USER_API_AUTH_HEADERS | APPLICATION_JSON_HEADERS)

    task = json.loads(response.text)

    """Step: Generate DataPoint"""
    data_point = fixtures.generate_data_point(task_id=task["task_id"])

    """Step: Call Data API to generate urls for uploading DataPoint's binary files and data."""
    file_names = list(data_point.binaries.keys())
    file_names.append('data_point.json')
    response = requests.post(url=f'{DATA_API_BASE_URI}/upload/{task["task_id"]}/{data_point.data_point_id}',
                             json=file_names,
                             headers={'task_api_key': task['client_api_key']} | APPLICATION_JSON_HEADERS)

    assert response.status_code == 200

    upload_urls = json.loads(response.text)
    for key, bytes_value in data_point.binaries.items():
        """Verify: urls for upload have been generated for all requested files."""
        assert key in upload_urls

        """Step: Upload binaries using generated upload urls """
        requests.post(url=upload_urls[key]['url_for_upload'],
                      data=bytes_value)
        data_point.file_uris.append(upload_urls[key]['url'])

    """Step: Upload DataPoint's data file."""
    data_point.binaries = None
    requests.post(url=upload_urls['data_point.json']['url_for_upload'],
                  data=json.dumps(data_point.dict(), indent=4))

    """Verify: download and verify uploaded binaries."""
    results: List[DownloadResult] = \
        await repository_async.download_batch_to_memory(data_point.file_uris)

    for result in results:
        assert result.status == '200'
        assert result.downloaded_bytes == fixtures.test_bytes

    """Verify: download and verify DataPoint's data."""
    download_result: DownloadResult = await repository_async.download_to_memory(upload_urls['data_point.json']['url'])
    assert download_result.status == '200'

    created_data_point = json.loads(download_result.downloaded_bytes.decode("utf-8"))
    assert created_data_point == data_point.dict()

    """Approach 1. [post] dataapi.xpl.ai/datapoints"""
    response = requests.post(url=f'{DATA_API_BASE_URI}/datapoints',
                             data=json.dumps([data_point.dict()], indent=4).encode('utf-8'),
                             headers={'task_api_key': task['client_api_key'], 'content-type': 'application/json'})

    assert response.status_code == 200
    """Approach 2. EVENTARC events
    eventarc events are received multiple times and were practically unusable."""
    # """Step: call Data API event handler to process uploaded DataPoint"""
    # data_point_file_bucket, data_point_file_name = \
    #     repository_async.parse_blob_uri(uri=upload_urls['data_point.json']['url'])
    # message = f"""{{
    #     "protoPayload":{{
    #         "resourceName":"projects/_/buckets/{data_point_file_bucket}/objects/{data_point_file_name}"
    #     }},
    #     "timestamp":"2021-07-07T22:54:34.358420863Z"
    # }}"""
    # response = requests.post(url=f'{DATA_API_BASE_URI}/storage/created',
    #                          data=message)

    # assert response.status_code == 204  # OK, no content

    """Approach 3. Pub/sub events. Not Implemented"""
    # message_data = {
    #     'name': data_point_file_name,
    #     'bucket': data_point_file_bucket
    # }
    # #
    # message_data_json = json.dumps(message_data)
    # message_data_bytes = message_data_json.encode()
    # message_data_base64_bytes = base64.b64encode(message_data_bytes)
    # message_data_base64_bytes_str = message_data_base64_bytes.decode()

    # response = requests.post(url=f'{DATA_API_BASE_URI}/storage/created',
    #                          json={'message': {
    #                                  'data': message_data_base64_bytes_str
    #                              }})

    # with open(os.path.join(os.path.dirname(__file__), 'event.json'), 'r') as file:
    #     event = file.read()
    #     response = requests.post(url=f'{DATA_API_BASE_URI}/storage/created',
    #                              data=event)
    #

# @pytest.mark.asyncio
# async def test__problematic_location():
#     """Step: call Data API event handler to process uploaded DataPoint"""
#     # data_point_file_bucket = 'xplai-datasets-41db42cbbeb749daa52c998df5d13d52-europe-north1'
#     # data_point_file_name = '163d97ac02264d3f87afbbb437e38030/8/a/7/8a70eba66a0e410a85d0aab67e6aa8c2/data_point.json'
#     data_point_file_bucket = 'xplai-datasets-52bbc5a50caa4c03bbb85fe83b56a669-europe-north1'
#     data_point_file_name = 'efb9a4656dee4a0cb221cf5b27f32214/0/0/3/00359e719bec4fc5ada466d4c44c391c/data_point.json'
#     message = f"""{{
#             "protoPayload":{{
#                 "resourceName":"projects/_/buckets/{data_point_file_bucket}/objects/{data_point_file_name}"
#             }},
#             "timestamp":"2021-07-07T22:54:34.358420863Z"
#         }}"""
#     response = requests.post(url=f'{DATA_API_BASE_URI}/storage/created',
#                              data=message)
#
#     assert response.status_code == 204

#  TODO: Write tests for each modality and cases for corrupt data.
