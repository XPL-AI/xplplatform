from xpl.task.task_service import TaskService
from xpl.task.entities import Model, ModelComponent

sut_task_service = TaskService()


def test__update_task_active_model():
    task_id = '649d49039d0f44cf8181d48cecf64dda'
    image_rep_url = 'gs://xplai-models-dev-europe-west1-01/' \
                    'b3830741a81f4a2b8523f46889dd2730/' \
                    'execution_2021-07-13_11-58-21_42907e61-130c-4cfe-af93-1e47874d2738/' \
                    'model_2021-07-13_11-59-12_37b5acf9-bccf-4f89-9579-a9a1ec3a9b45/image_rep.pts'

    test_road_signs_url = 'gs://xplai-models-dev-europe-west1-01/' \
                          'b3830741a81f4a2b8523f46889dd2730/' \
                          'execution_2021-07-13_11-58-21_42907e61-130c-4cfe-af93-1e47874d2738/' \
                          'model_2021-07-13_11-59-12_37b5acf9-bccf-4f89-9579-a9a1ec3a9b45/detection.pts'

    model = Model(
        model_id='model_82938485293',
        components={'image_rep.pts': ModelComponent(name='image_rep.pts',
                                                    url=image_rep_url),
                    'test_road_signs.pts': ModelComponent(name='test_road_signs.pt',
                                                          url=test_road_signs_url)
                    },
        output={'0': 'xpl:user:a3394c1bd66e'},
        version=2
    )

    sut_task_service.update_task_active_model(task_id=task_id,
                                              model=model)

    task = sut_task_service.get_task(task_id=task_id)

    assert task.model is not None
    assert task.model.model_id == model.model_id
    assert task.model.components is not None
    assert 'image_rep.pts' in task.model.components
    assert 'test_road_signs.pts' in task.model.components

    assert task.model.output is not None
    assert '0' in task.model.output
    assert task.model.output['0'] == 'xpl:user:a3394c1bd66e'

    assert task.model.version == 2
