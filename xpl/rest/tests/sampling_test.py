import json
from multiprocessing import Pool

from os.path import isfile, join
import requests
import unittest

from test_config import *


def _upload(upload: [(str, str)]):
    with open(upload[0], 'rb') as f:
        data = f.read()
        requests.put(upload[1], data=data)


class SamplingTests(unittest.TestCase):

    def test_upload_sample(self):
        headers = {'ClientKey': '204a24293e37496aa', 'ClientSecret': '204a24293e37496aa204a24293e37496aa'}
        file_names = sorted([f for f in os.listdir(SAMPLES_DIR) if isfile(join(SAMPLES_DIR, f)) and f != '.DS_Store'])
        model_id = '0e706893-c8aa-40bf-88d6-38d63a6380fb'

        files = []

        file_abs_names = [os.path.join(SAMPLES_DIR, n) for n in file_names]
        file_sizes = [os.path.getsize(n) for n in file_abs_names]

        try:
            for i in range(2):
                if file_names[i].endswith('jpg'):
                    file_stream = open(os.path.join(SAMPLES_DIR, file_names[i]), 'rb')
                    files.append(('files', file_stream))

            files.append(('files', open(os.path.join(SAMPLES_DIR, 'predictions.json'), 'rb')))

            """"Subject of test"""
            http_response = requests.post(f'{BASE_URL}/{model_id}/sample', files=files, headers=headers)
        finally:
            for file in files:
                file[1].close()

        upload_response = json.loads(json.loads(http_response.text))

        """"Validate"""
        self.assertIsNotNone(10, len(upload_response['sampleId']))

        sample_id = upload_response['sampleId']
        get_samples_response = requests.get(f"{BASE_URL}/{model_id}/sample/{sample_id}", headers=headers)
        uploaded_samples = json.loads(get_samples_response.text)

        self.assertEqual(3, len(uploaded_samples))
        self.assertTrue(f'{model_id}/{sample_id}/predictions.json' in uploaded_samples)

        """Validate each file size"""
        for idx, sample in enumerate(uploaded_samples.values()):
            response = requests.get(url=sample)
            self.assertEqual(file_sizes[idx], len(response.content))

    def test_upload_sample_over_url(self):
        model_id = '0e706893-c8aa-40bf-88d6-38d63a6380fb'

        url = f'{BASE_URL}/{model_id}/sample_upload_url'
        headers = {'ClientKey': '204a24293e37496aa', 'ClientSecret': '204a24293e37496aa204a24293e37496aa'}

        file_names = sorted([f for f in os.listdir(SAMPLES_DIR) if isfile(join(SAMPLES_DIR, f)) and f != '.DS_Store'])
        file_abs_names = [os.path.join(SAMPLES_DIR, n) for n in file_names]
        file_sizes = [os.path.getsize(n) for n in file_abs_names]

        http_response = requests.post(url, data=json.dumps({'sample_sizes': file_sizes}), headers=headers)
        self.assertEqual(200, http_response.status_code)

        upload_response = json.loads(json.loads(http_response.text))

        for idx, upload_url in enumerate(upload_response['uploadUrls']):
            with open(file_abs_names[idx], 'rb') as f:
                data = f.read()
                requests.put(upload_url, data=data)

        """Validate what was uploaded"""
        sample_id = upload_response['sampleId']
        get_samples_response = requests.get(f"{BASE_URL}/{model_id}/sample/{sample_id}", headers=headers)
        uploaded_samples = json.loads(get_samples_response.text)

        self.assertEqual(3, len(uploaded_samples))
        self.assertTrue(f'{model_id}/{sample_id}/predictions.json' in uploaded_samples)

        """Validate each file size"""
        for idx, sample in enumerate(uploaded_samples.values()):
            response = requests.get(url=sample)
            self.assertEqual(file_sizes[idx], len(response.content))

    def test_upload_sample_over_url_parallel_upload(self):
        model_id = '0e706893-c8aa-40bf-88d6-38d63a6380fb'

        url = f'{BASE_URL}/{model_id}/sample_upload_url'
        headers = {'ClientKey': '204a24293e37496aa', 'ClientSecret': '204a24293e37496aa204a24293e37496aa'}

        file_names = sorted([f for f in os.listdir(SAMPLES_DIR) if isfile(join(SAMPLES_DIR, f)) and f != '.DS_Store'])
        file_abs_names = [os.path.join(SAMPLES_DIR, n) for n in file_names]
        file_sizes = [os.path.getsize(n) for n in file_abs_names]

        http_response = requests.post(url, data=json.dumps({'sample_sizes': file_sizes}), headers=headers)
        upload_response = json.loads(json.loads(http_response.text))

        uploads = []
        for idx, upload_url in enumerate(upload_response['uploadUrls']):
            uploads.append((file_abs_names[idx], upload_url))

        try:
            pool = Pool(16)
            pool.map(_upload, uploads)
        finally:
            pool.close()
            pool.join()

        """Validate what was uploaded"""
        sample_id = upload_response['sampleId']
        get_samples_response = requests.get(f"{BASE_URL}/{model_id}/sample/{sample_id}", headers=headers)
        uploaded_samples = json.loads(get_samples_response.text)

        self.assertEqual(3, len(uploaded_samples))
        self.assertTrue(f'{model_id}/{sample_id}/predictions.json' in uploaded_samples)

        """Validate each file size"""
        for idx, sample in enumerate(uploaded_samples.values()):
            response = requests.get(url=sample)
            self.assertEqual(file_sizes[idx], len(response.content))


if __name__ == '__main__':
    unittest.main()
