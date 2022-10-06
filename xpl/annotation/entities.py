from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from xpl.concept.concept_service import Concept


# class InstanceFrame(BaseModel):
#     data_point_file: str
#     original_location: Optional[Dict]
#     original_text: Optional[str]
#     annotated_location: Optional[Dict]
#     annotated_text: Optional[str]
#
#
# class InstanceOfConcept(BaseModel):
#     instance_id: str
#     frames: List[InstanceFrame]
#
#
# class AnnotationJob(BaseModel):
#     user_id: str
#     task_id: str
#     model_id: str
#     concept: Concept
#     instances: Dict[str, InstanceOfConcept]


class SimpleInstanceFrame(BaseModel):
    row_id: str
    data_point_id: str
    instance_id: str
    data_point_file: str
    original_location: Optional[Dict]
    original_text: Optional[str]
    annotated_location: Optional[Dict]
    annotated_text: Optional[str]


class SimpleAnnotationJob(BaseModel):
    annotation_job_id: str
    user_id: str
    task_id: str
    model_id: str
    status: str
    concept: Concept
    concepts: Dict[str, Concept]
    instance_frames: Dict[str, SimpleInstanceFrame]


class AnnotationInstance(BaseModel):
    row_id: Optional[str]
    instance_id: str
    predictor_id: str
    concept_id: str
    original_location: Optional[Dict[str, Optional[float]]]
    original_text: Optional[str]
    annotated_location: Optional[Dict[str, Optional[float]]]
    annotated_text: Optional[str]


class AnnotationDataPoint(BaseModel):
    data_point_id: str
    data_point_file: str
    annotation_instances: Dict[str, AnnotationInstance]


class AnnotationJob(BaseModel):
    annotation_job_id: str
    annotation_job_bucket_name: Optional[str]
    annotation_job_file_name: Optional[str]
    user_id: str
    task_id: str
    concepts: Dict[str, Concept]
    data_points: Dict[str, AnnotationDataPoint]


class AnnotationJobInfo(BaseModel):
    annotation_job_id: str
    annotation_job_bucket_name: Optional[str]
    annotation_job_file_name: Optional[str]
    user_id: str
    task_id: str
    concepts: Dict[str, str]
    status: str
    data_points_count: Optional[int]
    instances_count: Optional[int]
    createdOn: datetime
    submittedOn: Optional[datetime]
