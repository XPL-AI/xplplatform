####################################################################################################
# File: image_dataset.py                                                                           #
# File Created: Wednesday, 14th July 2021 3:12:46 pm                                               #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Friday, 5th November 2021 10:47:27 am                                             #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################
import logging
from typing import Optional
from augment.data_augmenter import DataAugmenter
from augment.image_augmenter import ImageAugmenter
import numpy
import pandas
import torch
import torchvision
from xpl.dataset.dataset.xpl_dataset import XPLDataset
from xpl.dataset.augment.utils import augmenting_the_image, load_image_from_disk

logger = logging.getLogger(__name__)

class ImageDataset(XPLDataset):

    def __getitem__(self,
                    idx: int,
                    ) -> dict:
        # TODO: read from groupedby datapoint and not all data point
        data_point_id, indices_in_dataframe = self.grouped_data_points_list[idx]

        data_point_local_file_name = self.data_points['data_point_local_file'].values[indices_in_dataframe[0]]

        assert isinstance(data_point_local_file_name, str), '\n'.join(f'{data_point_local_file_name=}',
                                                                      f'{type(data_point_local_file_name)=}',
                                                                      f'{idx=}',
                                                                      f'{self.data_points.index[idx]=}')
        target_name = list(self.targets.keys())[0]

        targets = self.data_points[target_name].values[indices_in_dataframe]
        bounding_boxes = numpy.stack((self.data_points['center_x'].values[indices_in_dataframe],
                                      self.data_points['center_y'].values[indices_in_dataframe],
                                      self.data_points['half_width'].values[indices_in_dataframe],
                                      self.data_points['half_height'].values[indices_in_dataframe])).T

        loaded_sample = self.augment(image_filename=data_point_local_file_name,
                                     targets=targets,
                                     bounding_boxes=bounding_boxes)

        if loaded_sample is None:
            self.report_sample_is_corrupt(index=idx)
            data_point = {f'index': None,
                          f'{self.input_name}': None,
                          f'{target_name}_label': None,
                          f'{target_name}_objectiveness': None,
                          f'{target_name}_locations': None,
                          f'{target_name}_transform_matrix': None,
                          f'{target_name}_image_size': None,
                          }

        else:
            transformed_image, objectiveness, locations, labels, transform_matrix, image_size = loaded_sample
            data_point = {f'index': self.data_points.index[idx],
                          f'{self.input_name}': transformed_image,
                          f'{target_name}_label': labels,
                          f'{target_name}_objectiveness': objectiveness,
                          f'{target_name}_locations': locations,
                          f'{target_name}_transform_matrix': transform_matrix,
                          f'{target_name}_image_size': image_size,
                          }
        return data_point

    def init_data_augmenter(self,
                            input_size: dict[str, int],
                            output_size: dict[str, int],
                            background_data_points: Optional[pandas.DataFrame],
                            ) -> ImageAugmenter:
        # TODO fix annotation type based on datapoints
        if self.input_set_name == 'train':
            augmentation_score = .1
            return ImageAugmenter(augmentation_score=augmentation_score,
                                  input_size=(input_size['input_width'],
                                              input_size['input_height']),
                                  output_size=(output_size['target_width'],
                                               output_size['target_height']),
                                  background_data_points=background_data_points)

        else:
            return ImageAugmenter(augmentation_score=0,
                                  input_size=(input_size['input_width'],
                                              input_size['input_height']),
                                  output_size=(output_size['target_width'],
                                               output_size['target_height']),
                                  background_data_points=None)

    def calculate_informativeness(self,
                                  measurement: dict,
                                  sample: dict
                                  ) -> float:
        return 0.0
