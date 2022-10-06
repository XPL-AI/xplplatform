import unittest
import json

import requests

from test_config import *


class TaskTestCases(unittest.TestCase):
    def test_create_task_and_generate_model__200_AccuracyTest(self):
        task = """
        {
            "customerId":"204a24293e37496aa",
            "name":"TennisShot",
            "modality":"video",
            "concepts":
                [
                    { "name": "backhand" },
                    { "name": "forehand" },
                    { "name": "serve" },
                    { "name": "volley" }
                ],
            "size":"xl"
        }
        """

        """Subject of test"""
        HEADERS['content-type'] = 'application/json'
        http_response = requests.post(f'{BASE_URL}/task', data=task, headers=HEADERS)
        self.assertEqual(200, http_response.status_code)

        task = json.loads(http_response.text)
        print(task)

        """Validate"""
        self.assertEqual(HEADERS['ClientKey'], task['customerId'])
        self.assertEqual('TennisShot', task['name'])
        self.assertEqual('video', task['modality'])
        self.assertEqual('xl', task['size'])
        self.assertEqual(4, len(task['concepts']))

        self.assertTrue('modelId' in task)
        self.assertIsNotNone(task['modelId'])

        task_id = task['taskId']

        http_response = requests.get(f'{BASE_URL}/task/{task_id}/model', headers=HEADERS)

        model = json.loads(json.loads(http_response.text))
        self.assertEqual(HEADERS['ClientKey'], model['customerId'])
        self.assertEqual('xl', model['size'])
        self.assertEqual('TennisShot', model['name'])
        self.assertEqual(task['modelId'], model['modelId'])

        self.assertTrue('input' in model)
        self.assertTrue('modality' in model['input'])
        self.assertTrue('resolution' in model['input'])
        self.assertEqual('video', model['input']['modality'])
        self.assertEqual('500x500', model['input']['resolution'])

        self.assertFalse('modelFileUri' in model)
        self.assertTrue('modelUrl' in model)

        self.assertTrue('output' in model)
        self.assertEqual(4, len(model['output']))

        self.assertEqual('backhand', model['output']['0']['name'])
        self.assertEqual('backhand', model['output']['0']['conceptId'])
        self.assertEqual('forehand', model['output']['1']['name'])
        self.assertEqual('forehand', model['output']['1']['conceptId'])
        self.assertEqual('serve', model['output']['2']['name'])
        self.assertEqual('serve', model['output']['2']['conceptId'])
        self.assertEqual('volley', model['output']['3']['name'])
        self.assertEqual('volley', model['output']['3']['conceptId'])

        self.assertTrue('annotationTask' in model)
        self.assertTrue('sampling' in model)
        self.assertTrue('format' in model['sampling'])
        self.assertTrue('storageBuckets' in model['sampling'])
        self.assertEqual('png', model['sampling']['format'])
        self.assertEqual(1, len(model['sampling']['storageBuckets']))
        self.assertEqual('xplai-samples-dev-europe-west1-01', model['sampling']['storageBuckets'][0])

    def test_create_task_from_existing_model__200_AccuracyTest(self):
        model_setup = compile_coc0_train17_model_output(init_keypoints=False)

        task = f"""
        {{
            "customerId":"204a24293e37496aa",
            "name":"PersonDetector_Resnet50_COCOTrain17",
            "modality":"video",
            "modelSetup":{{
                "name": "resnet50",
                "modelFileUri": "gs://xplai-models-dev-europe-west1-01/barebone/video/barebone_resnet50.pt",
                "outputs": {
                json.dumps(model_setup, indent=4)
                }
            }},
            "size":"xl"
        }}
        """

        HEADERS['content-type'] = 'application/json'
        """Subject of test"""
        http_response = requests.post(f'{BASE_URL}/task', data=task, headers=HEADERS)
        self.assertEqual(200, http_response.status_code)
        task = json.loads(http_response.text)
        print(task)
        """Validate"""
        self.assertEqual(HEADERS['ClientKey'], task['customerId'])
        self.assertEqual('PersonDetector_Resnet50_COCOTrain17', task['name'])
        self.assertEqual('video', task['modality'])

        # self.assertEqual(5+17, len(task['concepts']))
        self.assertEqual(6, len(task['concepts']))

        self.assertTrue('modelId' in task)
        self.assertIsNotNone(task['modelId'])

        task_id = task['taskId']

        http_response = requests.get(f'{BASE_URL}/task/{task_id}/model', headers=HEADERS)

        model = json.loads(json.loads(http_response.text))
        self.assertEqual(HEADERS['ClientKey'], model['customerId'])
        self.assertEqual('xl', model['size'])
        self.assertEqual('resnet50', model['name'])
        self.assertEqual(task['modelId'], model['modelId'])

        self.assertTrue('input' in model)
        self.assertTrue('modality' in model['input'])
        self.assertTrue('resolution' in model['input'])
        self.assertEqual('video', model['input']['modality'])
        self.assertEqual('500x500', model['input']['resolution'])

        self.assertFalse('modelFileUri' in model)
        self.assertTrue('modelUrl' in model)

        self.assertTrue('output' in model)
        # self.assertEqual(5+17, len(model['output']))
        self.assertEqual(6, len(model['output']))

        self.assertEqual('unlabeled', model['output']['0']['name'])
        self.assertEqual('unlabeled', model['output']['0']['conceptId'])
        self.assertEqual('person', model['output']['1']['name'])
        self.assertEqual('person', model['output']['1']['conceptId'])
        self.assertEqual('parking meter', model['output']['14']['name'])
        self.assertEqual('parking_meter', model['output']['14']['conceptId'])
        self.assertEqual('ceiling-tile', model['output']['103']['name'])
        self.assertEqual('ceiling_tile', model['output']['103']['conceptId'])

        # self.assertEqual('left_wrist', model['output']['1__10']['name'])
        # self.assertEqual('person__keypoint__left_wrist', model['output']['1__10']['conceptId'])

        self.assertTrue('annotationTask' in model)
        self.assertTrue('sampling' in model)
        self.assertTrue('format' in model['sampling'])
        self.assertTrue('storageBuckets' in model['sampling'])
        self.assertEqual('png', model['sampling']['format'])
        self.assertEqual(1, len(model['sampling']['storageBuckets']))
        self.assertEqual('xplai-samples-dev-europe-west1-01', model['sampling']['storageBuckets'][0])


def compile_coc0_train17_model_output(init_keypoints=False):
    model_setup = {}
    with open("data/COCOtrain2017/labels_short.txt") as file:
        for line in file:
            (key, label) = line.split(': ')
            label = label.replace('\n', '')
            concept_id = label.replace(' ', '_').replace('-', '_').lower()
            model_setup[str(key)] = {'name': label, 'conceptId': concept_id}

    if init_keypoints:
        person_key = 1
        with open("data/COCOtrain2017/person__keypoints_labels.txt") as file:
            for line in file:
                (key, label) = line.split(': ')
                label = label.replace('\n', '')
                concept_id = label.replace(' ', '_').replace('-', '_').lower()
                concept_id = f'person__keypoint__{concept_id}'

                model_setup[str(f'{person_key}__{key}')] = {'name': label, 'conceptId': concept_id}

    return model_setup


if __name__ == '__main__':
    unittest.main()
