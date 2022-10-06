from datetime import datetime
from typing import List
import uuid

from google.cloud import firestore

from xpl.data import DataService
from xpl.data.repositories import DataItem
from xpl.concept import concept_service

from xpl.annotation.entities import *

ANNOTATION_JOB_COLLECTION_NAME = 'annotation_jobs'
FIRESTORE_CLIENT = firestore.Client()


class AnnotationService:
    def __init__(self):
        global FIRESTORE_CLIENT
        if FIRESTORE_CLIENT is None:
            FIRESTORE_CLIENT = firestore.Client()
        self.__firestore_client = FIRESTORE_CLIENT

    def create_simple_annotation_jobs(self,
                                      user_id: str,
                                      task_id: str,
                                      model_id: str,
                                      data_row_ids: List[str],
                                      split_by_each_concepts: bool
                                      ) -> List[SimpleAnnotationJob]:
        data_service = DataService()
        data_items = data_service.load_dataset_by_row_ids(user_id=user_id,
                                                          task_id=task_id,
                                                          processed=False,
                                                          row_ids=data_row_ids)

        data_items.sort(key=lambda x: getattr(x, 'instance_id'))

        annotation_jobs: Dict[str, SimpleAnnotationJob] = {}
        for item in data_items:
            if item.concept_id in annotation_jobs:
                annotation_jobs[item.concept_id].instance_frames[item.row_id] = SimpleInstanceFrame(row_id=item.row_id,
                                                                                                    data_point_id=item.data_point_id,
                                                                                                    instance_id=item.instance_id,
                                                                                                    data_point_file=item.data_point_file,
                                                                                                    original_location=item.location,
                                                                                                    original_text=item.text)
            else:
                concept = concept_service.get_by_id(item.concept_id)
                annotation_job = SimpleAnnotationJob(annotation_job_id=uuid.uuid4().hex,
                                                     user_id=user_id,
                                                     task_id=task_id,
                                                     model_id=model_id,
                                                     concept=concept,
                                                     status='not_started',
                                                     instance_frames={item.row_id: SimpleInstanceFrame(row_id=item.row_id,
                                                                                                       data_point_id=item.data_point_id,
                                                                                                       instance_id=item.instance_id,
                                                                                                       data_point_file=item.data_point_file,
                                                                                                       original_location=item.location,
                                                                                                       original_text=item.text)})
                annotation_jobs[item.concept_id] = annotation_job

        for concept_id, annotation_job in annotation_jobs.items():
            self.__firestore_client \
                .collection(ANNOTATION_JOB_COLLECTION_NAME) \
                .document(annotation_job.annotation_job_id) \
                .set(annotation_job.dict())

        return list(annotation_jobs.values())

    def list_annotation_jobs(self,
                             task_id: str):
        collection_ref = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME)
        query_ref = collection_ref.where('task_id', u'==', task_id)

        documents = query_ref.stream()
        result: List[SimpleAnnotationJob] = []
        for doc in documents:
            result.append(SimpleAnnotationJob(**doc.to_dict()))

        return result

    def get_annotation_job(self,
                           annotation_job_id: str):
        task_doc = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME).document(annotation_job_id).get()
        if not task_doc.exists:
            raise JobNotFoundException(f'annotation_job_id {annotation_job_id} not found in collection={ANNOTATION_JOB_COLLECTION_NAME}')

        return SimpleAnnotationJob(**task_doc.to_dict())

    def submit_annotation_job(self,
                              simple_annotation_job: SimpleAnnotationJob):
        data_items: List[DataItem] = []
        stored_simple_annotation_job = self.get_annotation_job(annotation_job_id=simple_annotation_job.annotation_job_id)

        for key_row_id, instance_frame in stored_simple_annotation_job.instance_frames.items():
            data_item = DataItem(data_point_id=instance_frame.data_point_id,
                                 data_point_file=instance_frame.data_point_file,
                                 timestamp=datetime.utcnow(),
                                 concept_id=simple_annotation_job.concept.concept_id,
                                 instance_id=instance_frame.instance_id,
                                 predictor_type='annotator',
                                 predictor_id=simple_annotation_job.user_id,
                                 input_set='train',
                                 log_informativeness=0.0,
                                 logit_confidence=1.0,
                                 location=simple_annotation_job.instance_frames[key_row_id].annotated_location)
            data_items.append(data_item)
            instance_frame.annotated_location = simple_annotation_job.instance_frames[key_row_id].annotated_location

        data_service = DataService()
        data_service.ingest_data_items(user_id=simple_annotation_job.user_id,
                                       task_id=simple_annotation_job.task_id,
                                       processed=False,
                                       data_items=data_items)

        stored_simple_annotation_job.status = 'complete'
        self.__firestore_client \
            .collection(ANNOTATION_JOB_COLLECTION_NAME) \
            .document(stored_simple_annotation_job.annotation_job_id) \
            .set(stored_simple_annotation_job.dict())


class JobNotFoundException(Exception):
    pass
