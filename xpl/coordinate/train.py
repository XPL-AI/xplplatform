####################################################################################################
# File: train.py                                                                                   #
# File Created: Friday, 9th July 2021 1:56:27 pm                                                   #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Tuesday, 14th September 2021 5:09:33 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import argparse
from xpl.coordinate.training_coordinator import TrainingCoordinator


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-u', '--user', type=str)
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-f', '--few_shot', default=False, action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    training_coordinator = TrainingCoordinator(user_name=arguments.user,
                                               task_name=arguments.task,
                                               few_shot_learning=arguments.few_shot)
    training_coordinator.start_training()
