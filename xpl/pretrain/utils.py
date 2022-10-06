####################################################################################################
# File: copy_state_dict.py                                                                         #
# File Created: Wednesday, 21st July 2021 4:08:00 pm                                               #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Monday, 20th September 2021 1:07:50 pm                                            #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import math
import matplotlib
import torch
import numpy
from difflib import SequenceMatcher
import matplotlib.pyplot as plt


from xpl.model.model_service import ModelService
from xpl.task.task_service import TaskService, Task
from xpl.user.user_service import UserService
from xpl.model.neural_net.xpl_model import XPLModel

splitter = '__xpl__'


def resolve_user_name_and_task_name(user_name: str,
                                    task_name: str
                                    ) -> Task:
    task_service = TaskService()
    user_service = UserService()
    user = user_service.resolve_user(username=user_name)
    user_tasks = task_service.list_user_tasks(user_id=user.user_id)
    tasks_matching_name = []
    for task in user_tasks:
        if task.name and task.name.lower().strip() == task_name:
            tasks_matching_name.append(task)

    assert len(tasks_matching_name) == 1, f'Fatal error, \n{tasks_matching_name=}\n is not unique for \n{user=}'

    task = tasks_matching_name[0]
    return task


def name_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_best_matchs(similarity_matrix,
                     min_i: int,
                     min_j: int,
                     max_i: int,
                     max_j: int,
                     ) -> list[tuple[int, int]]:
    print('-->', min_i, min_j, max_i, max_j)
    if max_i <= min_i or max_j <= min_j:
        return []

    sub_matrix = similarity_matrix[min_i:max_i, min_j:max_j]
    (i, j) = numpy.unravel_index(numpy.argmax(sub_matrix, axis=None), sub_matrix.shape)
    i += min_i
    j += min_j
    return find_best_matchs(similarity_matrix, min_i, min_j, i-1, j-1) + \
        [(i, j)] +\
        find_best_matchs(similarity_matrix, i+1, j+1, max_i, max_j)


def flatten(nested_dict):
    flat_dict = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            for nested_k, final_v in flatten(v).items():
                flat_dict[f'{k}{splitter}{nested_k}'] = final_v
        else:
            flat_dict[k] = v
    return [(k, v) for k, v in flat_dict.items()]


def get_state_dict(pretrained_state_dict, xpl_neural_net_state_dict):

    flat_pretrained_state_dict = flatten(pretrained_state_dict)
    flat_xpl_state_dict = flatten(xpl_neural_net_state_dict)

    similarity_matrix = numpy.zeros((len(flat_xpl_state_dict),
                                    len(flat_pretrained_state_dict)))

    for xpl_index, (xpl_k, xpl_weight) in enumerate(flat_xpl_state_dict):
        for pretrained_index, (pretrained_k, pretrained_weight) in enumerate(flat_pretrained_state_dict):
            if xpl_weight.size() == pretrained_weight.size():
                similarity_matrix[xpl_index, pretrained_index] = \
                    name_similarity(pretrained_k, xpl_k) + math.sqrt(1+torch.numel(xpl_weight))

    plt.title('This must be a diagonal matrix!!!')
    plt.imshow(similarity_matrix)
    plt.show()

    return_state_dict = {}
    for i in range(len(flat_xpl_state_dict)):
        return_state_dict[flat_xpl_state_dict[i][0]] = flat_pretrained_state_dict[i][1]

    return return_state_dict


def almost_equal(tensor_a, tensor_b, eps=1e-5):
    tensor_a = tensor_a.cpu()
    tensor_b = tensor_b.cpu()

    if not tensor_a.size() == tensor_b.size():
        print(f'{tensor_a.shape=} is not equal {tensor_b.shape=}')
        return False
    if (tensor_a.mean() - tensor_b.mean()).abs() > eps:
        print(f'{tensor_a.mean()=} is not equal {tensor_b.mean()=}')
        return False
    if (tensor_a.std() - tensor_b.min()).std() > eps:
        print(f'{tensor_a.std()=} is not equal {tensor_b.std()=}')
        return False

    if (tensor_a.max() - tensor_b.max()).abs() > eps:
        print(f'{tensor_a.max()=} is not equal {tensor_b.max()=}')
        return False
    if (tensor_a.min() - tensor_b.min()).abs() > eps:
        print(f'{tensor_a.min()=} is not equal {tensor_b.min()=}')
        return False

    if (tensor_a - tensor_b).abs().max() > eps:
        print(f'{tensor_a.max()=}, {tensor_a.min()=}')
        print(f'{tensor_b.max()=}, {tensor_b.min()=}')
        print(f'{(tensor_a - tensor_b).abs().max()} is greater than {eps}')
        return False

    print(f'tensors are equal')
    return True


def update_pretrained_on_server(models: dict[str, XPLModel],
                                modality: str,
                                model_size: str,
                                ) -> None:

    task = resolve_user_name_and_task_name(user_name='pretrain@xpl.ai',
                                           task_name=f'{modality}_background')
    model_service = ModelService()

    model_service.save_pretrained_model(user_id=task.user_id,
                                        task_id=task.task_id,
                                        models=models,
                                        modality=modality,
                                        model_size=model_size)
