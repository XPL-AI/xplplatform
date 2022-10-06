import json
import requests

from xpl.rest.tests import config


BASE_URI = config["userapi_uri"]


def test__login():
    url = f'{BASE_URI}/login'
    username = email = '8c6a5c56a2@test.com'
    secret = ''

    data = {
        'username': username,
        'secret': secret
    }

    response = requests.post(url=url, json=data)

    assert 200 == response.status_code
    assert secret == json.loads(response.text)['api_key']


def test__post_task__401():
    headers = {'username': '4e020bc2b7@test.com',
               'api_key': ''}

    task_definition = '{ "modality": "image", "name": "", "csv": "test" }'

    response = requests.post(url=f'{BASE_URI}/task',
                             headers=headers,
                             data=task_definition)

    assert 401 == response.status_code
