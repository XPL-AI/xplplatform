# import unittest
# import json
#
# import uuid
#
# import requests
#
# from test_config import *
#
#
# class ConceptTestCases(unittest.TestCase):
#     def test_new_concept(self):
#         rand_id = str(uuid.uuid4())[:8]
#         concept_id = f'AUTOTEST_concept_id_{rand_id}'
#         name = f'AUTOTEST_name_{rand_id}'
#         bnId = f'AUTOTEST_bnId_{rand_id}'
#
#         concept_setup = f'''{{
#             "conceptId": "{concept_id}",
#             "name": "{name}",
#             "bnId": "{bnId}"
#         }}
#         '''
#
#         """Subject of test"""
#         http_response = requests.post(f'{BASE_URL}/concept', data=concept_setup, headers=HEADERS)
#         concept = json.loads(http_response.text)
#         self.assertEqual(concept_id, concept['conceptId'])
#         self.assertEqual(name, concept['name'])
#         self.assertEqual(bnId, concept['bnId'])
#         # self.assertEqual(concept_id + '_c', concept['data']['table'])
#
#         """Subject of test"""
#         http_response = requests.get(f'{BASE_URL}/concept/{concept_id}', headers=HEADERS)
#         concept = json.loads(http_response.text)
#
#         self.assertEqual(concept_id, concept['conceptId'])
#         self.assertEqual(name, concept['name'])
#         self.assertEqual(bnId, concept['bnId'])
#         # self.assertEqual(concept_id + '_c', concept['data']['table'])
#
#     def test_new_concept__concept_id_not_given__200_id_picked_from_name(self):
#         rand_id = str(uuid.uuid4())[:8]
#         name = f'AUTOTEST_name_{rand_id}'
#         bnId = f'AUTOTEST_bnId_{rand_id}'
#
#         concept_setup = f'''{{
#                     "name": "{name}",
#                     "bnId": "{bnId}"
#                 }}
#                 '''
#
#         """Subject of test"""
#         http_response = requests.post(f'{BASE_URL}/concept', data=concept_setup, headers=HEADERS)
#         concept = json.loads(http_response.text)
#         self.assertEqual(name, concept['conceptId'])
#         self.assertEqual(name, concept['name'])
#         self.assertEqual(bnId, concept['bnId'])
#         # self.assertEqual(name + '_c', concept['data']['table'])
#
#         """Subject of test"""
#         http_response = requests.get(f'{BASE_URL}/concept/{name}', headers=HEADERS)
#         concept = json.loads(http_response.text)
#
#         self.assertEqual(name, concept['conceptId'])
#         self.assertEqual(name, concept['name'])
#         self.assertEqual(bnId, concept['bnId'])
#         # self.assertEqual(name + '_c', concept['data']['table'])
#
#     def test_new_concept__bnId_not_given__200_bnId_is_None(self):
#         rand_id = str(uuid.uuid4())[:8]
#         name = f'AUTOTEST_name_{rand_id}'
#
#         concept_setup = f'''{{
#                     "name": "{name}"
#                 }}
#                 '''
#
#         """Subject of test"""
#         http_response = requests.post(f'{BASE_URL}/concept', data=concept_setup, headers=HEADERS)
#         concept = json.loads(http_response.text)
#         self.assertEqual(None, concept['bnId'])
#
#         """Subject of test"""
#         http_response = requests.get(f'{BASE_URL}/concept/{name}', headers=HEADERS)
#         concept = json.loads(http_response.text)
#
#         self.assertEqual(None, concept['bnId'])
#
#     def test_search_concept(self):
#         rand_id = str(uuid.uuid4())[:8]
#         concept_id = f'AUTOTEST_concept_id_{rand_id}'
#         name = f'AUTOTEST_name_{rand_id}'
#         bnId = f'AUTOTEST_bnId_{rand_id}'
#
#         concept_setup = f'''{{
#                     "conceptId": "{concept_id}",
#                     "name": "{name}",
#                     "bnId": "{bnId}"
#                 }}
#                 '''
#
#         """Create Concept"""
#         http_response = requests.post(f'{BASE_URL}/concept', data=concept_setup, headers=HEADERS)
#         concept = json.loads(http_response.text)
#
#         """Get concept py id"""
#         http_response = requests.get(f'{BASE_URL}/concept/{name}', headers=HEADERS)
#         concept = json.loads(http_response.text)
#
#         """Search concept"""
#         http_response = requests.get(f'{BASE_URL}/concepts/search?field=bnId&value={bnId}', headers=HEADERS)
#         concept = json.loads(http_response.text)[0]
#
#         self.assertEqual(bnId, concept['bnId'])
