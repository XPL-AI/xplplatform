import asyncio
import json
import uuid

from datetime import datetime
from time import perf_counter
from typing import Dict, List

from google.cloud import firestore
from google.cloud.firestore_v1 import CollectionReference, DocumentReference

from xpl.infrastructure.storage import CloudStorageRepository, repository_async

from xpl.data import DataService
from xpl.data.repositories import DataItem
from xpl.concept import concept_service

from xpl.annotation.entities import AnnotationJob, AnnotationDataPoint, AnnotationInstance, Concept, AnnotationJobInfo

ANNOTATION_JOB_COLLECTION_NAME = 'annotation_jobs'
FIRESTORE_CLIENT = firestore.Client()


class AnnotationService2:
    def __init__(self):
        global FIRESTORE_CLIENT
        if FIRESTORE_CLIENT is None:
            FIRESTORE_CLIENT = firestore.Client()
        self.__firestore_client = FIRESTORE_CLIENT
        self.__storage = CloudStorageRepository()
        self.__data_service = DataService()

    def create_annotation_jobs(self,
                               user_id: str,
                               task_id: str,
                               model_id: str,
                               concepts_ids: List[str],
                               row_ids: List[str],
                               dataset_bucket_name: str
                               ) -> List[AnnotationJob]:
        data_service = DataService()
        data_items = data_service.load_dataset_by_row_ids(user_id=user_id,
                                                          task_id=task_id,
                                                          processed=False,
                                                          row_ids=row_ids)

        concepts_dict = self.__load_concepts_dict(concepts_ids, data_items)

        data_points_dict: Dict[str, List[DataItem]] = {}
        for data_item in data_items:
            if data_item.data_point_id in data_points_dict:
                data_points_dict[data_item.data_point_id].append(data_item)
            else:
                data_points_dict[data_item.data_point_id] = [data_item]

        data_point_id_batches = repository_async.make_batches(list(data_points_dict.keys()))

        jobs = []
        for data_point_id_batch in data_point_id_batches:
            job = AnnotationJob(annotation_job_id=uuid.uuid4().hex,
                                user_id=user_id,
                                task_id=task_id,
                                concepts=concepts_dict,
                                data_points={})

            for data_point_id in data_point_id_batch:
                data_items_in_data_point = data_points_dict[data_point_id]
                data_point = AnnotationDataPoint(data_point_id=data_point_id,
                                                 data_point_file=data_items_in_data_point[0].data_point_file,
                                                 annotation_instances={})
                for data_item in data_items_in_data_point:
                    annotation_item = AnnotationInstance(row_id=data_item.row_id,
                                                         instance_id=data_item.instance_id,
                                                         predictor_id=data_item.predictor_id,
                                                         concept_id=data_item.concept_id,
                                                         original_location=data_item.location,
                                                         original_text=data_item.text,
                                                         annotated_location=data_item.location,
                                                         annotated_text=data_item.text)
                    data_point.annotation_instances[data_item.instance_id] = annotation_item

                job.data_points[data_point.data_point_id] = data_point
            jobs.append(job)

        loop = asyncio.get_event_loop()
        jobs = loop.run_until_complete(self.__upload_jobs_to_storage(jobs=jobs,
                                                                     task_id=task_id,
                                                                     dataset_bucket_name=dataset_bucket_name))
        self.__insert_job_info(jobs)

        return jobs

    def load_annotation_job(self,
                            annotation_job_id: str):
        job_info = self.get_annotation_job_by_id(annotation_job_id)
        job_bytes: bytes = self.__storage.download_as_bytes(blob_name=job_info.annotation_job_file_name,
                                                            bucket_name=job_info.annotation_job_bucket_name)

        job_dict = json.loads(job_bytes)
        return AnnotationJob(**job_dict)

    async def save_data_point(self,
                              data_point: AnnotationDataPoint,
                              annotation_job_id: str):
        job = self.load_annotation_job(annotation_job_id)

        # override temporary URL that was received for download purpose
        data_point.data_point_file = job.data_points[data_point.data_point_id].data_point_file
        job.data_points[data_point.data_point_id] = data_point

        await self.__update_jobs_in_storage([job])

    def list_annotation_jobs_for_user(self,
                                      user_id: str
                                      ) -> List[AnnotationJobInfo]:
        collection = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME)
        per_user_collection = collection.document('per_user').collection('jobs')
        document_ref: DocumentReference = per_user_collection.document(user_id)
        doc = document_ref.get()

        if doc.exists:
            jobs: List[AnnotationJobInfo] = []
            for job in doc.to_dict().values():
                j = AnnotationJobInfo(**job)
                jobs.append(j)
            return jobs
        else:
            return []

    def list_annotation_jobs_for_task(self,
                                      task_id: str
                                      ) -> List[AnnotationJobInfo]:
        collection = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME)
        per_task_collection = collection.document('per_task').collection('jobs')
        document_ref: DocumentReference = per_task_collection.document(task_id)
        doc = document_ref.get()

        if doc.exists:
            jobs: List[AnnotationJobInfo] = []
            for job in doc.to_dict().values():
                j = AnnotationJobInfo(**job)
                jobs.append(j)
            return jobs
        else:
            return []

    def get_annotation_job_by_id(self,
                                 annotation_job_id: str
                                 ) -> AnnotationJobInfo:
        collection = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME)
        per_job_collection = collection.document('per_job').collection('jobs')
        document_ref: DocumentReference = per_job_collection.document(annotation_job_id)
        doc = document_ref.get()

        if doc.exists:
            return AnnotationJobInfo(**doc.to_dict())
        else:
            raise JobNotFoundException(f'Annotation job {annotation_job_id=} does not exist')

    def submit_annotation_job(self,
                              annotation_job_id: str):
        job = self.load_annotation_job(annotation_job_id=annotation_job_id)
        data_items: List[DataItem] = []

        for data_point_id, data_point in job.data_points.items():
            for instance_id, annotation_instance in data_point.annotation_instances.items():
                data_item = DataItem(data_point_id=data_point.data_point_id,
                                     data_point_file=data_point.data_point_file,
                                     timestamp=datetime.utcnow(),
                                     concept_id=annotation_instance.concept_id,
                                     instance_id=annotation_instance.instance_id,
                                     predictor_type='annotator',
                                     predictor_id=job.user_id,
                                     input_set='train',
                                     log_informativeness=0.0,
                                     logit_confidence=1.0,
                                     location=annotation_instance.annotated_location,
                                     text=annotation_instance.annotated_text)
                data_items.append(data_item)

        self.__data_service.ingest_data_items(user_id=job.user_id,
                                              task_id=job.task_id,
                                              processed=False,
                                              data_items=data_items)

        self.__update_job_info_status(annotation_job_id=job.annotation_job_id,
                                      user_id=job.user_id,
                                      task_id=job.task_id,
                                      new_status='complete')

    def __load_concepts_dict(self,
                             concept_ids,
                             data_items:
                             List[DataItem]
                             ) -> Dict[str, Concept]:
        concept_set = set(i.concept_id for i in data_items)
        concept_set.update(concept_ids)

        concepts_dict: Dict[str, Concept] = dict.fromkeys(concept_set, None)
        for concept_id in concept_set:
            concepts_dict[concept_id] = concept_service.get_by_id(concept_id)

        return concepts_dict

    async def __upload_jobs_to_storage(self,
                                       jobs: List[AnnotationJob],
                                       dataset_bucket_name: str,
                                       task_id: str
                                       ) -> List[AnnotationJob]:

        uploads: List[repository_async.UploadResource] = []
        perf = {}
        t1_start = perf_counter()
        for job in jobs:
            annotation_job_file_name = f'{task_id}/annotation_jobs/job__{self.__get_timestamp()}__{job.annotation_job_id}.json'
            job.annotation_job_file_name = annotation_job_file_name
            job.annotation_job_bucket_name = dataset_bucket_name
            upload_resource = repository_async.UploadResource(bucket_name=dataset_bucket_name,
                                                              blob_name=annotation_job_file_name,
                                                              bytes_to_upload=job.json().encode('utf-8'))
            uploads.append(upload_resource)
        perf['prepare uploads'] = perf_counter() - t1_start
        t1_start = perf_counter()
        await repository_async.upload_batch(uploads)

        perf['upload_batch_wrapper'] = perf_counter() - t1_start
        return jobs

    async def __update_jobs_in_storage(self,
                                       jobs: List[AnnotationJob]
                                       ) -> List[AnnotationJob]:

        uploads: List[repository_async.UploadResource] = []
        perf = {}
        t1_start = perf_counter()
        for job in jobs:
            annotation_job_file_name = job.annotation_job_file_name
            upload_resource = repository_async.UploadResource(bucket_name=job.annotation_job_bucket_name,
                                                              blob_name=annotation_job_file_name,
                                                              bytes_to_upload=job.json().encode('utf-8'))
            uploads.append(upload_resource)
        perf['prepare uploads'] = perf_counter() - t1_start
        t1_start = perf_counter()
        await repository_async.upload_batch(uploads)

        perf['upload_batch_wrapper'] = perf_counter() - t1_start
        return jobs

    def __insert_job_info(self,
                          jobs: List[AnnotationJob]):
        collection = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME)
        # collection.document('per_user').set(document_data={}, merge=True)
        # collection.document('per_task').set(document_data={}, merge=True)
        # collection.document('per_job').set(document_data={}, merge=True)

        per_user_collection: CollectionReference = collection.document('per_user').collection('jobs')
        per_task_collection: CollectionReference = collection.document('per_task').collection('jobs')
        per_job_collection: CollectionReference = collection.document('per_job').collection('jobs')

        for job in jobs:
            instances_count = 0
            for id, data_point in job.data_points.items():
                instances_count += len(data_point.annotation_instances)
            job_info: AnnotationJobInfo = AnnotationJobInfo(
                annotation_job_id=job.annotation_job_id,
                annotation_job_bucket_name=job.annotation_job_bucket_name,
                annotation_job_file_name=job.annotation_job_file_name,
                user_id=job.user_id,
                task_id=job.task_id,
                concepts={key: job.concepts[key].name for key in job.concepts.keys()},
                status='new',
                data_points_count=len(job.data_points),
                instances_count=instances_count,
                createdOn=datetime.utcnow()
            )

            per_user_collection.document(job.user_id).set({job.annotation_job_id: job_info.dict()}, merge=True)
            per_task_collection.document(job.task_id).set({job.annotation_job_id: job_info.dict()}, merge=True)
            per_job_collection.add(job_info.dict(), job.annotation_job_id)

    def __update_job_info_status(self,
                                 annotation_job_id: str,
                                 user_id: str,
                                 task_id: str,
                                 new_status: str):
        collection = self.__firestore_client.collection(ANNOTATION_JOB_COLLECTION_NAME)
        per_user_collection: CollectionReference = collection.document('per_user').collection('jobs')
        per_task_collection: CollectionReference = collection.document('per_task').collection('jobs')
        per_job_collection: CollectionReference = collection.document('per_job').collection('jobs')

        doc_ref = per_user_collection.document(user_id)
        doc_ref.update({f'{annotation_job_id}.status': new_status})

        doc_ref = per_task_collection.document(task_id)
        doc_ref.update({f'{annotation_job_id}.status': new_status})

        doc_ref = per_job_collection.document(annotation_job_id)
        doc_ref.update({'status': new_status, 'submittedOn': datetime.utcnow()})

    def __get_timestamp(self
                        ) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


class JobNotFoundException(Exception):
    pass
