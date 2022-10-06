import json

from typing import List

from xpl.annotation import AnnotationService, SimpleAnnotationJob, SimpleInstanceFrame, Job, AnnotationService2

sut_annotation_service = AnnotationService()
sut_annotation_service2 = AnnotationService2()


def make_batches(strings: List[str], n=100):
    return [strings[i * n:(i + 1) * n] for i in range((len(strings) + n - 1) // n)]


def test__create_simple_annotation_jobs():
    with open('row_ids.txt') as f:
        rows = f.read().splitlines()

    batches = make_batches(rows, n=250)

    for batch in batches:
        annotation_jobs: List[SimpleAnnotationJob] = sut_annotation_service.create_simple_annotation_jobs(
            user_id='9713ef454935409f80c640aacc97cbbf',
            task_id='409d412f1e644a36aeaae1082ad76c03',
            model_id='jyfuhjgjh',
            data_row_ids=batch)
        assert len(annotation_jobs) > 0



