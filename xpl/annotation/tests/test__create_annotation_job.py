import json
from time import perf_counter

from typing import List

from xpl.annotation import AnnotationService2, AnnotationJob

sut_annotation_service = AnnotationService2()


def test__create_annotation_jobs():
    user_id = '460ddd1e9dc84d16bffbff902e363010'
    task_id = '9f5e5bfbd8ff40aea2a6d4d5930cbc41'
    dataset_bucket_name = 'xplai-datasets-460ddd1e9dc84d16bffbff902e363010-europe-north1'

    with open('row_ids.txt') as f:
        rows = f.read().splitlines()

    perf = {}
    t1_start = perf_counter()
    annotation_jobs: List[AnnotationJob] = sut_annotation_service.create_annotation_jobs(
        user_id=user_id,
        task_id=task_id,
        model_id='jyfuhjgjh',
        row_ids=rows,
        dataset_bucket_name=dataset_bucket_name,
        concepts_ids=['xpl:user:ffcd264e7aa3', 'xpl:user:f4fe69f71dd4', 'xpl:user:6d7e1bf306dc', 'xpl:user:6e45e431ba00']
    )

    perf['create_annotation_jobs'] = perf_counter() - t1_start
    t1_start = perf_counter()

    # for job in annotation_jobs:
    #     with open(f'job_{job.annotation_job_id}.json', 'w') as f:
    #         job_json = job.json()
    #         f.write(job_json)
    # assert len(annotation_jobs) > 0
    #
    # perf['write_to_files'] = perf_counter() - t1_start
    # t1_start = perf_counter()

    user_jobs = sut_annotation_service.list_annotation_jobs_for_user(user_id)

    perf['list_annotation_jobs_for_user'] = perf_counter() - t1_start
    t1_start = perf_counter()

    task_jobs = sut_annotation_service.list_annotation_jobs_for_task(task_id)

    perf['list_annotation_jobs_for_task'] = perf_counter() - t1_start
    t1_start = perf_counter()

    job_id = task_jobs[0].annotation_job_id
    single_job_info = sut_annotation_service.get_annotation_job_by_id(job_id)

    perf['get_annotation_job_by_id'] = perf_counter() - t1_start
    t1_start = perf_counter()

    job = sut_annotation_service.load_annotation_job(single_job_info.annotation_job_id)

    perf['load_annotation_job'] = perf_counter() - t1_start
    t1_start = perf_counter()

    data_points_list = list(job.data_points.values())
    data_point = data_points_list[0]
    annotation_item = list(data_point.annotation_instances.values())[0]

    annotation_item.annotated_location = {"center_x": 0.6520833333333333,
                                          "half_height": 0.05027901785714287,
                                          "center_y": 0.6468191964285714,
                                          "half_width": 0.09666666666666662}
    t1_start = perf_counter()
    sut_annotation_service.save_data_point(data_point, single_job_info.annotation_job_id)
    perf['save_data_point'] = perf_counter() - t1_start
    t1_start = perf_counter()

    # t1_start = perf_counter()
    sut_annotation_service.submit_annotation_job(single_job_info.annotation_job_id)
    # perf['submit_annotation_job'] = perf_counter() - t1_start
    t1_start = perf_counter()
