####################################################################################################
# File: data_augmenter.py                                                                          #
# File Created: Monday, 13th September 2021 11:00:41 am                                            #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Tuesday, 14th September 2021 4:54:21 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


import abc

class DataAugmenter:
    
    @abc.abstractmethod
    def __call__(self):
        pass