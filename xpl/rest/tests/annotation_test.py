import unittest
import uuid
from datetime import datetime
import requests

from test_config import *


class AnnotationTestCase(unittest.TestCase):
    def test_post_annotation(self):
        annotation = f"""{{
            "concept_id": "person",
            "owner_id": "test_owner",
            "sample_file_name": "testfilename_{uuid.uuid4()}",
            "instance_id": 1,
            "source": "test_source",
            "source_id": "test_source_id",
            "timestamp":  "{str(datetime.utcnow())}",
            "location": {{
                "x1":0.3,
                "y1":0.4,
                "x2":0.9,
                "y2":0.8
            }}
        }}"""

        """Subject of test"""
        http_response = requests.post(f'{BASE_URL}/annotation', data=annotation, headers=HEADERS)

        self.assertEqual(200, http_response.status_code)


if __name__ == '__main__':
    unittest.main()
