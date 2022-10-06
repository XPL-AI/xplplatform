import json
import requests

from xpl.rest.tests import config

BASE_URI = config["userapi_uri"]
AUTH_HEADERS = {'username': config["valid_username"], 'api_key': config["valid_api_key"]}
APPLICATION_JSON_HEADERS = {'content-type': 'application/json'}


def test__post_task():
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

    response = requests.post(url=f'{BASE_URI}/task',
                             data=data,
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS)

    assert response.status_code == 200
    task = json.loads(response.text)
    assert task is not None
    assert task['task_id'] is not None
    assert task['name'] == 'test_task'
    assert task['modality'] == 'video'
    assert task['model_size'] == 'xxxs'
    assert task['concepts'] is not None
    assert len(task['concepts']) == 3
    assert task['concepts'][0]['display_name'] == 'ONE'
    assert task['concepts'][1]['display_name'] == 'TWO'
    assert task['concepts'][2]['display_name'] == 'THREE'

    response = requests.get(url=f'{BASE_URI}/task/{task["task_id"]}',
                            headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS)

    assert response.status_code == 200
    task = json.loads(response.text)
    assert task is not None
    assert task['task_id'] is not None
    assert task['name'] == 'test_task'
    assert task['modality'] == 'video'
    assert task['model_size'] == 'xxxs'
    assert task['concepts'] is not None
    assert len(task['concepts']) == 3
    assert task['concepts'][0]['display_name'] == 'ONE'
    assert task['concepts'][1]['display_name'] == 'TWO'
    assert task['concepts'][2]['display_name'] == 'THREE'
    assert task['model'] is None

    response = requests.get(url=f'{BASE_URI}/tasks',
                            headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS)
    assert response.status_code == 200
    tasks = json.loads(response.text)
    assert len(tasks) > 0


def test__insufficient_input__422_and_message_information_is_appropriate():
    data = f"""
    {{
        "concept_definitions": [
            {{ "user_provided_concept_name": "ONE"}}
        ]
    }}
    """

    response = requests.post(url=f'{BASE_URI}/task',
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS,
                             data=data)

    assert 422 == response.status_code
    assert 'errors' in json.loads(response.text), 'Response body should contain json with "errors"'
    errors = json.loads(response.text)['errors']

    assert len(errors) == 2
    assert 'Task definition misses required argument: modality.' in errors[0]
    assert 'Supported modalities:' in errors[0]

    assert 'Task definition misses required argument: model_size.' == errors[1]


def test__modality_not_supported__422_message_information_is_appropriate():
    data = f"""
    {{        
        "modality": "text",
        "model_size": "xl",
        "concept_definitions": [
            {{ "user_provided_concept_name": "ONE"}}
        ]
    }}
    """

    response = requests.post(url=f'{BASE_URI}/task',
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS,
                             data=data)

    assert 422 == response.status_code
    assert 'errors' in json.loads(response.text), 'Response body should contain json with "errors"'
    errors = json.loads(response.text)['errors']

    assert len(errors) == 1
    assert 'Modality: "text" is not a supported modality.' in errors[0]


def test__model_size_for_modality_not_supported__422_message_information_is_appropriate():
    data = f"""
    {{        
        "modality": "video",
        "model_size": "xl",
        "concept_definitions": [
            {{ "user_provided_concept_name": "ONE"}}
        ]
    }}
    """

    response = requests.post(url=f'{BASE_URI}/task',
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS,
                             data=data)

    assert response.status_code == 422
    assert 'errors' in json.loads(response.text), 'Response body should contain json with "errors"'
    errors = json.loads(response.text)['errors']

    assert len(errors) == 1
    assert 'model_size "xl" is not available for modality "video"' in errors[0]


def test__invalid_concept_id__422_message_information_is_appropriate():
    data = f"""
    {{        
        "modality": "video",
        "model_size": "xxxs",
        "concept_definitions": [
            {{ "user_provided_concept_name":"ONE", "concept_id": "invalid_concept_id" }}
        ]
    }}
    """

    response = requests.post(url=f'{BASE_URI}/task',
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS,
                             data=data)

    assert response.status_code == 422
    assert 'errors' in json.loads(response.text), 'Response body should contain json with "errors"'
    errors = json.loads(response.text)['errors']

    assert len(errors) == 1
    assert 'Concept concept_id=invalid_concept_id does not exist.' in errors[0]


def test__invalid_wn_id__422_message_information_is_appropriate():
    data = f"""
    {{        
        "modality": "video",
        "model_size": "xxxs",
        "concept_definitions": [
            {{ "user_provided_concept_name":"ONE", "wn_id": "invalid_wn_id" }}
        ]
    }}
    """

    response = requests.post(url=f'{BASE_URI}/task',
                             headers=AUTH_HEADERS | APPLICATION_JSON_HEADERS,
                             data=data)

    assert response.status_code == 422
    assert 'errors' in json.loads(response.text), 'Response body should contain json with "errors"'
    errors = json.loads(response.text)['errors']

    assert len(errors) == 1
    assert 'Concept with wn_id="invalid_wn_id" does not exist.' in errors[0]
