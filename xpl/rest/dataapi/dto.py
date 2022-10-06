from pydantic import BaseModel
from typing import Optional, Dict, List


class DataItem(BaseModel):
    concept_id: str
    instance_id: str
    predictor_type: str
    predictor_id: str
    input_set: str
    log_informativeness: float
    logit_confidence: float
    location: Optional[Dict[str, Optional[float]]]
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
