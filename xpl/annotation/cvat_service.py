import json

from typing import List

import requests

from xpl.annotation import config

from xpl.infrastructure.storage import CloudStorageRepository

CVAT_URL = config['cvat_url']
CVAT_AUTHORIZATION_TOKEN = config['cvat_authorization_token']


class CvatService:
    def create_annotation_project(self,
                                  project_name: str,
                                  labels: List[str]):
        labels = [{'name': lb, 'attributes': []} for lb in labels]
        p = {'name': project_name, 'labels': labels}

        headers = {'Authorization': f'Token {CVAT_AUTHORIZATION_TOKEN}', 'Content-type': 'application/json'}
        reply = requests.post(f'{CVAT_URL}/projects', data=json.dumps(p), headers=headers)

        if reply.status_code not in [201, 200]:
            raise Exception(f'Failed to setup CVAT project. http_status_code={reply.status_code}', f'body={reply.text}')

        project = json.loads(reply.text)
        project['url'] = project['url'].replace('/api/v1/projects', '/projects')
        return project


    def create_annotation_task(project_id: int,
                               name: str):
        task = {
            'name': name,
            'labels': [],
            'project_id': project_id
        }

        headers = {'Authorization': f'Token {CVAT_AUTHORIZATION_TOKEN}', 'Content-type': 'application/json'}
        reply = requests.post(f'{CVAT_URL}/tasks', data=json.dumps(task), headers=headers)

        if reply.status_code not in [200, 201]:
            raise Exception(f'Failed to setup CVAT annotation task. http_status_code={reply.status_code}',
                            f'body={reply.text}')

        return json.loads(reply.text)

    def upload_samples_to_annotation_task(task_id: int,
                                          samples_uri_list: [str],
                                          prediction_urls: [str] = None):
        samples = {}
        storage = CloudStorageRepository()
        for idx, uri in enumerate(samples_uri_list):
            content = storage.download_as_bytes_by_uri(uri=uri)
            media_type = uri.split('.')[-1]
            samples[f'client_files[{idx}]'] = (uri, content, f'image/{media_type}')

        samples['image_quality'] = (None, 70)
        samples['use_zip_chunks'] = (None, 'true')
        samples['use_cache'] = (None, 'true')

        headers = {'Authorization': f'Token {CVAT_AUTHORIZATION_TOKEN}'}
        reply = requests.post(f'{CVAT_URL}/tasks/{task_id}/data', files=samples, headers=headers)

        if reply.status_code not in [200, 201, 202]:
            raise Exception(f'Failed to setup CVAT annotation task. http_status_code={reply.status_code}',
                            f'body={reply.text}')

        return json.loads(reply.text)


    annotations = """
    {
      "version": 0,
       "tags": [
      ],
      "shapes": [
        {
          "type": "rectangle",
          "occluded": true,
          "z_order": 0,
          "points": [
            10, 10, 20, 20
          ],
          "id": 200,
          "frame": 0,
          "label_id": 27,
          "group": 0,
          "source": "string",
          "attributes": [
          ]
        },
        {
          "type": "rectangle",
          "occluded": true,
          "z_order": 0,
          "points": [
            10, 10, 20, 20
          ],
          "id": 201,
          "frame": 1,
          "label_id": 27,
          "group": 0,
          "source": "string",
          "attributes": [
          ]
        }],
        "tracks":[]
    }"""

    # def upload_initial_predictions(task_id):
    #     headers = {'Authorization': f'Token {CVAT_AUTHORIZATION_TOKEN}', 'Content-type': 'application/json'}
    #     reply = requests.put(f'{CVAT_URL}/tasks/{task_id}/annotations', data=annotations, headers=headers)
    #
    #     print(reply.text)
