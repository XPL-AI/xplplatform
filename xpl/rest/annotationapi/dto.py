from typing import Optional, Dict

from pydantic import BaseModel


class SimpleInstanceFrame(BaseModel):
    row_id: str
    data_point_file_url: str
    original_location: Optional[Dict]
    annotated_location: Optional[Dict]


class SimpleAnnotationJob(BaseModel):
    annotation_job_id: str
    user_id: str
    task_id: str
    model_id: str
    status: str
    concept_name: str
    instance_frames: Dict[str, SimpleInstanceFrame]
