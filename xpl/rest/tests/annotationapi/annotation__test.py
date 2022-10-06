import json
import requests

from xpl.rest.tests import config

BASE_URI = config["annotation_uri"]
AUTH_HEADERS = {'username': config["valid_username"], 'api_key': config["valid_api_key"]}
APPLICATION_JSON_HEADERS = {'content-type': 'application/json'}


def test__submit_annotation_job():
    annotation_job_id = '4ac42cd4ef8e4bbc973e57a53b069ddf'
    response = requests.get(url=f'{BASE_URI}/job/{annotation_job_id}',
                            headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS)

    annotation_job = json.loads(response.text)

    for key_row_id, instance_frame in annotation_job['instance_frames'].items():
        instance_frame['annotated_location'] = {"center_x": 0.4,
                                                "half_width": 0.4,
                                                "center_y": 0.4,
                                                "half_height": 0.4}

    response = requests.post(url=f'{BASE_URI}/job/{annotation_job_id}',
                             data=json.dumps(annotation_job, indent=4),
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS)

    assert response.status_code == 200
