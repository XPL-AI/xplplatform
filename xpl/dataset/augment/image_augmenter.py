####################################################################################################
# File: image_adjuster.py                                                                          #
# File Created: Tuesday, 7th September 2021 7:54:11 pm                                             #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Thursday, 11th November 2021 2:37:09 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Optional, Tuple, Union
import math
import numpy
import pandas
import torch

import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from matplotlib import patches
from xpl.dataset.augment.image_tools import fix_dimension_and_normalize, load_image
from xpl.dataset.augment.image_tools import get_random_transformed_image, convert_original_cxcy_to_cropped_cxcy

from xpl.dataset.augment.data_augmenter import DataAugmenter


class ImageAugmenter(DataAugmenter):

    def __init__(self,
                 augmentation_score: int,     # 0 means no augmentation, 1 means massive augmentation
                 input_size: Union[int, Tuple[int, int]],
                 output_size: Union[int, Tuple[int, int]],
                 background_data_points: Optional[pandas.DataFrame] = None,
                 ) -> None:
        assert augmentation_score >= 0 and augmentation_score <= 1, f'{augmentation_score=} must be between 0 to 10'
        self.__augmentation_score = augmentation_score
        self.__image_size = input_size if isinstance(input_size, (list, tuple)) else (input_size, input_size)
        self.__output_size = output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size)
        self.__has_background = False
        if background_data_points is not None:
            self.__has_background = True
            self.__background_data_frame = background_data_points

        self.noise_augmenter = self.__get_noise_model(self.__augmentation_score)

    def __call__(self,
                 image_filename: str,
                 targets: numpy.array,
                 bounding_boxes: numpy.array,
                 visualize: bool = False,
                 ) -> Tuple[torch.Tensor,  # input : C x input_h x input_w
                            torch.Tensor,  # objectiveness: target_h x target_w
                            torch.Tensor,  # bounding_boxes: target_h x target_w x N x 4
                            torch.Tensor,  # targets: target_h x target_w x N
                            numpy.ndarray,  # transform_matrix: 3 x 3
                            numpy.ndarray,  # original_image_size: 2
                            ]:

        image, success = load_image(image_filename)
        if not success:
            print(image_filename)
            return None

        # If the image is too big, downscale it to sqrt(2) times our desired image size
        # Otherwise, we take care of resizing in the augmentation process
        scale = min(int(min(self.__image_size) * 1.4142),  # sqrt(2)
                    min(image.shape[0], image.shape[1]))

        image, original_image_size = fix_dimension_and_normalize(input_image=image,
                                                                 scale_to=scale,
                                                                 keep_aspect=True,
                                                                 colorspace='RGB',
                                                                 )

        transformed_image, transform_matrix, image_size = get_random_transformed_image(image,
                                                                                       self.__image_size,
                                                                                       self.__augmentation_score,
                                                                                       )

        transformed_bounding_boxes = convert_original_cxcy_to_cropped_cxcy(transform_matrix,
                                                                           bounding_boxes,
                                                                           original_image_size=image_size,
                                                                           cropped_image_size=self.__image_size)

        transformed_bounding_boxes = numpy.nan_to_num(transformed_bounding_boxes, nan=0.5)

        valid_mask = numpy.logical_and(
            numpy.logical_and(transformed_bounding_boxes[:, 0] > 0,
                              transformed_bounding_boxes[:, 0] < 1),
            numpy.logical_and(transformed_bounding_boxes[:, 1] > 0,
                              transformed_bounding_boxes[:, 1] < 1))
        targets = targets[valid_mask]
        transformed_bounding_boxes = transformed_bounding_boxes[valid_mask, :]

        num_objects = transformed_bounding_boxes.shape[0]

        h, w = self.__output_size
        objectiveness = torch.zeros(h, w, dtype=torch.long)
        locations = torch.zeros(h, w, num_objects + 1, 4)
        labels = torch.zeros(h, w, num_objects + 1, dtype=torch.long)

        if num_objects > 0:

            bounding_box_area = transformed_bounding_boxes[:, 2] * transformed_bounding_boxes[:, 3]
            sorted_indices = numpy.argsort(bounding_box_area)[::-1]
            transformed_bounding_boxes = transformed_bounding_boxes[sorted_indices, :]
            targets = targets[sorted_indices]

            for i, (image_cx, image_cy, rw, rh) in enumerate(transformed_bounding_boxes):
                cell_w_index = int(image_cx * w)
                cell_h_index = int(image_cy * h)

                relative_x = image_cx * w - int(image_cx * w)
                relative_y = image_cy * h - int(image_cy * h)

                current_objects_in_this_cell = objectiveness[cell_h_index,
                                                             cell_w_index]
                locations[cell_h_index, cell_w_index, current_objects_in_this_cell, 0] = relative_x
                locations[cell_h_index, cell_w_index, current_objects_in_this_cell, 1] = relative_y
                locations[cell_h_index, cell_w_index, current_objects_in_this_cell, 2] = numpy.log(w*rw + 1e-8)
                locations[cell_h_index, cell_w_index, current_objects_in_this_cell, 3] = numpy.log(h*rh + 1e-8)
                labels[cell_h_index, cell_w_index, current_objects_in_this_cell] = targets[i]
                objectiveness[cell_h_index, cell_w_index] += 1

            max_num_objects_per_cell = objectiveness.max()
            locations = locations[:, :, :max_num_objects_per_cell, :]
            labels = labels[:, :, :max_num_objects_per_cell]
            #objectiveness = objectiveness > 0

        transformed_image = self.noise_augmenter(image=transformed_image)
        transformed_image = torch.FloatTensor(transformed_image).permute((2, 0, 1)).clamp(max=1, min=0)
        

        if visualize:
            self.__visualize(image=transformed_image,
                             locations=locations,
                             targets=targets,
                             objectiveness=objectiveness)

        return (transformed_image,
                objectiveness,
                locations,
                labels,
                transform_matrix,
                image_size)

    def __get_noise_model(self,
                          augmentation_score
                          ) -> iaa.Sequential:
        return iaa.Sequential([iaa.Sometimes(augmentation_score / 2,
                                             iaa.GaussianBlur(sigma=(0, augmentation_score))
                                             ),
                               iaa.Sometimes(augmentation_score / 2,
                                             iaa.LinearContrast((1 - augmentation_score/2,
                                                                 1 + augmentation_score/2))),
                               iaa.Sometimes(augmentation_score / 2,
                                             iaa.AdditiveLaplaceNoise(scale=(0, augmentation_score / 2))
                                             ),
                               iaa.Sometimes(augmentation_score / 2,
                                             iaa.AdditiveGaussianNoise(loc=0,
                                                                       scale=(0.0, augmentation_score / 2),
                                                                       per_channel=augmentation_score)
                                             ),
                               iaa.Sometimes(augmentation_score / 2,
                                             iaa.Multiply((1 - augmentation_score/2,
                                                           1 + augmentation_score/2),
                                                          per_channel=augmentation_score)
                                             ),
                               ],
                              random_order=True,)

    def __visualize(self,
                    image,
                    locations,
                    targets,
                    objectiveness,
                    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='auto')
        ax.imshow(image.permute((1, 2, 0)).numpy())
        image_h = image.shape[1]
        image_w = image.shape[2]
        h = locations.shape[0]
        w = locations.shape[1]

        for h_ind in range(h):
            for w_ind in range(w):
                if objectiveness[h_ind, w_ind] > 0:
                    for n in range(locations[h_ind, w_ind].shape[0]):
                        (cx, cy, rw, rh) = locations[h_ind, w_ind, n]
                        if cx + cy + rw + rh == 0:
                            continue

                        new_cx = (w_ind + cx) / w 
                        new_cy = (h_ind + cy) / h
                        new_rw = math.exp(rw) / w
                        new_rh = math.exp(rh) / h

                        ann = patches.Ellipse((new_cx * image_w, new_cy * image_h),
                                              2 * new_rw * image_w,
                                              2 * new_rh * image_h,
                                              angle=0,
                                              linewidth=2,
                                              fill=False,
                                              zorder=2,
                                              color='RED')
                        ax.add_patch(ann)
        plt.show()


if __name__ == '__main__':
    import os
    filename = os.path.join(os.environ['XPL_CODE_DIR'],
                            'data/cache',
                            'xplai-datasets-4df5d87d6e8b4cb993c3f54d14f6feb5-europe-north1',
                            '0b06dd38a37a4becbcda051248d08f9a',
                            '1/0/8/1083a299f26540b0aa80b3d04a1febb8',
                            'input.jpg')

    targets = numpy.array([30, 30, 30, 50, 50, 50, 50, -1, 77])
    bboxes = numpy.array([[0.67770312, 0.83765625, 0.01321875, 0.00582292],
                          [0.72310156, 0.83830208, 0.01664844, 0.01069792],
                          [0.63161719, 0.83468750, 0.01277344, 0.00481250],
                          [0.33531250, 0.65038542, 0.03487500, 0.08513542],
                          [0.37752344, 0.61235417, 0.07078906, 0.14718750],
                          [0.56950000, 0.62327083, 0.08807812, 0.15160417],
                          [0.63967188, 0.62085417, 0.05428125, 0.13710417],
                          [numpy.nan,  numpy.nan, numpy.nan, numpy.nan ],
                          [0.89657812, 0.58006250, 0.10342187, 0.17733333]])

    augmenter = ImageAugmenter(augmentation_score=.5,
                               input_size=(512, 320),
                               output_size=(32, 20),
                               background_data_points=None)

    augmenter(image_filename=filename,
              targets=targets,
              bounding_boxes=bboxes,
              visualize=True)
