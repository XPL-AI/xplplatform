import uuid

from pydantic import BaseModel
from typing import Dict, List, Optional


class DataItem(BaseModel):
    concept_id: str
    instance_id: str
    predictor_type: str
    predictor_id: str
    input_set: str
    log_informativeness: float
    logit_confidence: float
    location: Optional[Dict[str, float]]
    text: Optional[str]
    value: Optional[float]


class DataPoint(BaseModel):
    task_id: str
    data_point_id: str
    binaries: Optional[Dict[str, bytes]]
    file_uris: List[str]
    data_items: List[DataItem]
    previous_data_point_id: Optional[str]
    next_data_point_id: Optional[str]
    collected_by_device_fingerprint_id: Optional[str]


test_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx' \
             b'\xdac\xfc\xff\x9f\xa1\x1e\x00\x07\x82\x02\x7f=\xc8H\xef\x00\x00\x00\x00IEND\xaeB`\x82'


def generate_data_point(task_id: str
                        ) -> DataPoint:
    return DataPoint(
        task_id=task_id,
        data_point_id=uuid.uuid4().hex,
        binaries={'input.png': test_bytes,
                  'layer1.tensor': test_bytes},
        file_uris=[],
        collected_by_device_fingerprint_id=str(uuid.getnode()),
        data_items=[
            DataItem(
                concept_id='xpl:wn:n7127006',
                instance_id=uuid.uuid4().hex,
                predictor_type='model',
                predictor_id='model_id1_v1',
                input_set='',
                log_informativeness=0.2,
                logit_confidence=1,
                location={'center_x': 0.5, 'half_width': 0.5},
                text=f'text{task_id}',
                value=0.25
            ),
            DataItem(
                concept_id='xpl:wn:n7130341',
                instance_id=uuid.uuid4().hex,
                predictor_type='model',
                predictor_id='model_id1_v1',
                input_set='',
                log_informativeness=0.2,
                logit_confidence=1,
                location={'center_x': 0.5, 'half_width': 0.5}
            )
        ]
    )
