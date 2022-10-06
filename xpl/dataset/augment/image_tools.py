####################################################################################################
# File: image_tools.py                                                                             #
# File Created: Wednesday, 13th October 2021 10:25:30 am                                           #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 10th November 2021 2:14:27 pm                                          #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import os
import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import patches


backend = os.environ.get('IMAGE_BACKEND', 'cv2').lower()
if backend == 'cv2':
    import cv2
elif backend == 'skimage':
    from skimage import io
    from skimage.transform import warp, AffineTransform, resize


def get_random_transformed_image(input_image: numpy.array,
                                 output_size,
                                 augmentation_score,
                                 ):
    input_size = numpy.array(input_image.shape)
    transform_matrix = get_random_spatial_transform_matrix(input_size, output_size, augmentation_score)
    if backend == 'cv2':
        transformed_image = cv2.warpPerspective(input_image, transform_matrix, (int(output_size[1]), int(output_size[0])))
    if backend == 'skimage':
        transform = AffineTransform(transform_matrix)
        transformed_image = warp(input_image, transform.inverse, output_shape=(int(output_size[0]), int(output_size[1])))

    if transformed_image.ndim == 2:
        transformed_image = transformed_image.reshape([*transformed_image.shape, 1])

    return transformed_image, transform_matrix, input_size


def get_fixed_transformed_image(input_image,
                                output_size,
                                augmentation_score,
                                ):
    input_size = numpy.array(input_image.shape)
    transform_matrix = get_fixed_spatial_transform_matrix(input_size, output_size, augmentation_score)
    if backend == 'cv2':
        transformed_image = cv2.warpPerspective(input_image, transform_matrix, (int(output_size[1]), int(output_size[0])))
    if backend == 'skimage':
        transform = AffineTransform(transform_matrix)
        transformed_image = warp(input_image, transform.inverse, output_shape=(int(output_size[0]), int(output_size[1])))
    if transformed_image.ndim == 2:
        transformed_image = transformed_image.reshape([*transformed_image.shape, 1])
    return transformed_image, transform_matrix, input_size


def get_inverse_transformed_image(transformed_image,
                                  input_size,
                                  transform_matrix,
                                  ):
    if backend == 'cv2':
        input_image = cv2.warpPerspective(src=transformed_image,
                                          transform_matrix=transform_matrix,
                                          dsize=(int(input_size[1]), int(input_size[0]))
                                          )
    if backend == 'skimage':
        transform = AffineTransform(matrix=numpy.linalg.inv(transform_matrix))
        input_image = warp(image=transformed_image,
                           inverse_map=transform.inverse,
                           output_shape=(int(input_size[0]), int(input_size[1])))
    return input_image


def get_random_spatial_transform_matrix(input_size,
                                        output_size,
                                        augmentation_score=0,
                                        ):
    # With deep learning, data augmentation increases the accuracy of the model,
    # which is why we like to have as few copies of the same sample as possible
    # during the traning process. Hence we ignore the augmentation_score

    # If -in the future- it is discovered that we should augment the data
    # during the traning, we can use augmentation_score accordingly.
    rotation = 0 if input_size[0] > input_size[1] else 90

    if numpy.random.rand() < augmentation_score:
        rotation += numpy.random.normal(0, 45 * augmentation_score)

    return get_transform_matrix(input_size=input_size,
                                output_size=output_size,
                                center_ratio=numpy.random.normal(0, .5 * augmentation_score, 2),
                                rotation=rotation,
                                stretch=numpy.random.normal(loc=1, scale=augmentation_score/5, size=2),
                                # Mirror horizontally slightly less than random as we want
                                # the system to know left-right from the direction
                                horizontal_mirror=numpy.random.rand() > .5 if augmentation_score > 0 else 0,
                                vertical_mirror=numpy.random.rand() > .5 if augmentation_score > 0 else 0)


def get_fixed_spatial_transform_matrix(input_size, output_size, augmentation_score=0):
    fix_param = get_fix_spatial_transform_params(augmentation_score)
    return get_transform_matrix(input_size=input_size,
                                output_size=output_size,
                                center_ratio=fix_param['center_ratio'],
                                rotation=fix_param['rotation'],
                                stretch=fix_param['stretch'],
                                horizontal_mirror=fix_param['horizontal_mirror'],
                                vertical_mirror=fix_param['vertical_mirror'])


def get_fix_spatial_transform_params(augmentation_score):
    # good number for augmentations are :
    # 1: the image itself
    # 4: images flipped horizontally and vertically
    # 12: the 4 images are rotated -10, 0 and 10 degrees
    # 108: corner cropped of those 12 images
    center_ratio = [0, -1 / 4, 1 / 4]
    rotation = [0, -10, 10]
    horizontal_mirror = [False, True]
    vertical_mirror = [False, True]
    cnt = 0
    while True:
        for c_x in center_ratio:
            for c_y in center_ratio:
                for r in rotation:
                    for h in horizontal_mirror:
                        for v in vertical_mirror:
                            cnt += 1
                            if augmentation_score < cnt:
                                return {
                                    'center_ratio': (c_x, c_y),
                                    'stretch': (1 - abs(c_x), 1 - abs(c_y)),
                                    'rotation': r,
                                    'horizontal_mirror': h,
                                    'vertical_mirror': v
                                }
        # if we reach to this stage, it means our augmentation size has been
        # bigger than 108, so we make the cropped images smaller and the angle of rotations more:
        rotation = [r * 1.5 for r in rotation]
        center_ratio = [c * .1 for c in center_ratio]
        # and we go back to the "while True" loop to sample another 108 images.
        # Honestly, I don't think we will ever need to augment and image more
        # than 12 times but I just add support for infinite data augmentation


def get_transform_matrix(input_size,
                         output_size,
                         center_ratio=(0, 0),
                         rotation=0,
                         stretch=(1, 1),
                         horizontal_mirror=False,
                         vertical_mirror=False,
                         ):

    input_corners = numpy.array([[0, 0, 1],
                                 [input_size[1], 0, 1],
                                 [input_size[1], input_size[0], 1],
                                 [0, input_size[0], 1]],
                                dtype='float32').T
    output_corners = numpy.array([[0, 0, 1],
                                  [output_size[1], 0, 1],
                                  [output_size[1], output_size[0], 1],
                                  [0, output_size[0], 1]],
                                 dtype='float32').T

    input_center = numpy.mean(input_corners, axis=1)
    output_center_in_original_image = (numpy.array([*center_ratio, 0]) + 1) * input_center
    output_center_in_cropped_image = numpy.mean(output_corners, axis=1)

    output_corners_in_original_image = output_corners.copy()
    output_corners_in_original_image -= output_center_in_cropped_image[:, numpy.newaxis]
    output_corners_in_original_image *= numpy.array([*stretch, 0])[:, numpy.newaxis]
    output_corners_in_original_image[2, :] = 1
    output_corners_in_original_image = numpy.dot(rotate(rotation),
                                                 output_corners_in_original_image)
    output_corners_in_original_image += output_center_in_original_image[:, numpy.newaxis]
    output_corners_in_original_image[2, :] = 1

    if (horizontal_mirror):
        output_corners = output_corners[:, [1, 0, 3, 2]]
    if (vertical_mirror):
        output_corners = output_corners[:, [3, 2, 1, 0]]

    # numpy can solve AX=B, here we want to solve transform_matrix . original = cropped
    # To do so, we must reformulate the equation:
    # (transform_matrix . original)' = cropped' # => original' . transform_matrix' = cropped'

    transform_matrix = numpy.linalg.lstsq(output_corners_in_original_image.T,
                                          output_corners.T, rcond=-1)[0].T
    transform_matrix[numpy.abs(transform_matrix) < 1e-4] = 0  # because it is visually more appealing!

    return transform_matrix


def convert_original_cxcy_to_cropped_cxcy(transform_matrix,
                                          original_cxcy: numpy.array,
                                          original_image_size: tuple[int, int],
                                          cropped_image_size: tuple[int, int]
                                          ):
    cxcy_to_four_corner_transform = numpy.array([[1, 0, -1, 0],
                                                 [0, 1, 0, -1],
                                                 [1, 0, 1, 0],
                                                 [0, 1, 0, -1],
                                                 [1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [1, 0, -1, 0],
                                                 [0, 1, 0, 1]],
                                                dtype='float32').T
    four_corners = numpy.dot(original_cxcy, cxcy_to_four_corner_transform)
    cropped_four_corner = convert_original_points_to_cropped_image(transform_matrix,
                                                                   four_corners.reshape(-1, 2),
                                                                   original_image_size,
                                                                   cropped_image_size).reshape(-1, 8)

    avg_matrix = numpy.array([[1/4, 0, 1/4, 0, 1/4, 0, 1/4, 0],
                              [0, 1/4, 0, 1/4, 0, 1/4, 0, 1/4]]).T
    cropped_centers = numpy.dot(cropped_four_corner, avg_matrix)

    distances = numpy.abs(cropped_four_corner - numpy.tile(cropped_centers, [1, 4]))
    rxry = numpy.dot(distances, avg_matrix)

    cropped_cxcy = numpy.concatenate((cropped_centers, rxry), 1)

    return cropped_cxcy


def convert_cropped_cxcy_to_original_cxcy(transform_matrix,
                                          cropped_cxcy,
                                          original_image_size: tuple[int, int],
                                          cropped_image_size: tuple[int, int]
                                          ):
    cxcy_to_four_corner_transform = numpy.array([[1, 0, -1, 0],
                                                 [0, 1, 0, -1],
                                                 [1, 0, 1, 0],
                                                 [0, 1, 0, -1],
                                                 [1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [1, 0, -1, 0],
                                                 [0, 1, 0, 1]],
                                                dtype='float32').T
    four_corners = numpy.dot(cropped_cxcy, cxcy_to_four_corner_transform)
    original_four_corner = convert_cropped_image_points_to_original(transform_matrix,
                                                                    four_corners.reshape(-1, 2),
                                                                    original_image_size,
                                                                    cropped_image_size).reshape(-1, 8)

    avg_matrix = numpy.array([[1/4, 0, 1/4, 0, 1/4, 0, 1/4, 0],
                              [0, 1/4, 0, 1/4, 0, 1/4, 0, 1/4]]).T
    original_centers = numpy.dot(original_four_corner, avg_matrix)

    distances = numpy.abs(original_four_corner - numpy.tile(original_centers, [1, 4]))
    rxry = numpy.dot(distances, avg_matrix)

    original_cxcy = numpy.concatenate((original_centers, rxry), 1)

    return original_cxcy


def convert_original_points_to_cropped_image(transform_matrix,
                                             original_points,  # must be a numpy array of nx2 coordinates in relative values
                                             original_image_size: tuple[int, int],
                                             cropped_image_size: tuple[int, int]
                                             ):
    # points must be between 0 and 1, but they could rarely be outside of the image

    extra_ones = numpy.ones((original_points.shape[0], 1))
    original_points = numpy.concatenate((original_points, extra_ones), axis=1)

    original_points[:, 0] *= original_image_size[1]
    original_points[:, 1] *= original_image_size[0]

    transformed_points = numpy.dot(transform_matrix, original_points.T).T
    transformed_points /= numpy.expand_dims(transformed_points[:, 2], 1)
    transformed_points = transformed_points[:, 0:2]

    transformed_points[:, 0] /= cropped_image_size[1]
    transformed_points[:, 1] /= cropped_image_size[0]

    return transformed_points


def convert_cropped_image_points_to_original(transform_matrix,
                                             transformed_points,  # must be a numpy array of nx2 coordinates in relative values
                                             original_image_size: tuple[int, int],
                                             cropped_image_size: tuple[int, int]
                                             ):

    extra_ones = numpy.ones((transformed_points.shape[0], 1))
    transformed_points = numpy.concatenate((transformed_points, extra_ones), axis=1)

    transformed_points[:, 0] *= cropped_image_size[1]
    transformed_points[:, 1] *= cropped_image_size[0]

    original_points = numpy.dot(numpy.linalg.inv(transform_matrix), transformed_points.T).T
    original_points /= numpy.expand_dims(original_points[:, 2], 1)
    original_points = original_points[:, 0:2]

    original_points[:, 0] /= original_image_size[1]
    original_points[:, 1] /= original_image_size[0]

    return original_points


# I'm only using the rotate function at the moment, the rest are implemented inside
# get_transform_matrix. But I leave the functions here in case we later on decide to
# test something


def rotate(theta: float,
           transform_matrix: numpy.array = None,
           ):
    transform_matrix = transform_matrix if transform_matrix else numpy.eye(3, dtype='float32')
    theta = theta * math.pi / 180
    rt = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                      [numpy.sin(theta), numpy.cos(theta), 0],
                      [0, 0, 1]],
                     dtype='float32',
                     )
    rot_M = rt
    return numpy.dot(rot_M, transform_matrix)


def shear(l: float,
          transform_matrix: numpy.array = None,
          ):
    transform_matrix = transform_matrix if transform_matrix else numpy.eye(3, dtype='float32')
    rt = numpy.array([[1, l, 0],
                      [0, 1, 0],
                      [0, 0, 1]],
                     dtype='float32',
                     )
    rot_M = rt
    return numpy.dot(rot_M, transform_matrix)


def stretch(rho_x: float,
            rho_y: float,
            transform_matrix: numpy.array = None,
            ):
    transform_matrix = transform_matrix if transform_matrix else numpy.eye(3, dtype='float32')
    sc_M = numpy.array([[rho_x, 0, 0],
                        [0, rho_y, 0],
                        [0, 0, 1]],
                       dtype='float32')
    return numpy.dot(sc_M, transform_matrix)


def translate(x: float,
              y: float,
              transform_matrix: numpy.array = None,
              ) -> numpy.array:
    transform_matrix = transform_matrix if transform_matrix else numpy.eye(3, dtype='float32')
    tr_M = numpy.array([[1, 0, x],
                        [0, 1, y],
                        [0, 0, 1]],
                       dtype='float32')
    return numpy.dot(tr_M, transform_matrix)


def mirror(transform_matrix: numpy.array = None,
           ) -> numpy.array:
    transform_matrix = transform_matrix if transform_matrix else numpy.eye(3, dtype='float32')
    tr_M = numpy.array([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]],
                       dtype='float32')
    return numpy.dot(tr_M, transform_matrix)


# all images should be float, with the dimension: w x h x c
# where either w or h or both are equal to the "scale_to" config


def fix_dimension_and_normalize(input_image: numpy.array,
                                scale_to: int,
                                keep_aspect=True,
                                colorspace: str = 'RGB',
                                ):
    # add extra dimension if it doesn't have any
    if input_image.ndim == 2:
        input_image = input_image.reshape([*input_image.shape, 1])

    original_image_size = numpy.array(input_image.shape)

    # First, we resize the image
    if (keep_aspect):
        scale_size = (scale_to, scale_to)
    else:
        min_shape = min(input_image.shape[0], input_image.shape[1])
        scale_size = (int(scale_to * input_image.shape[0] / min_shape),
                      int(scale_to * input_image.shape[1] / min_shape))

    if backend == 'cv2':
        input_image = cv2.resize(input_image, scale_size, interpolation=cv2.INTER_LINEAR)
    if backend == 'skimage':
        input_image = resize(input_image, scale_size, order=1, mode='constant', anti_aliasing=False)

    # Then we convert the image so that the values be in the range of [0-1]
    if input_image.dtype in ['uint16', 'uint8', 'int16', 'int8']:
        max_val = numpy.iinfo(input_image.dtype).max
        min_val = numpy.iinfo(input_image.dtype).min
        input_image = input_image.astype('float32')
        input_image = (input_image - min_val) / (max_val - min_val + 1e-18)

    elif input_image.dtype in ['float32', 'float64']:
        assert(input_image.max() <= 1. and input_image.min() >= .0),\
            "after resizing the image with skimage, the max and min are %f,%f" % (
                input_image.max(), input_image.min())
    else:
        raise BaseException('Unknown image format type: "%s"' % (input_image.dtype))

    # Now we normalize the image
    #input_image = input_image * 2 - 1

    # at this point, image must be in the shape of [W, H, C]
    # removing the alpha channel:
    if input_image.shape[2] == 4:
        input_image = input_image[:, :, :3]

    # fix colorspace:
    if input_image.shape[2] == 1 and colorspace.lower() == 'RGB'.lower():
        input_image = numpy.repeat(input_image, 3, axis=2)
    elif input_image.shape[2] == 3 and colorspace.lower() == 'Gray'.lower():
        input_image = 0.2989 * input_image[:, :, 0] + 0.5870 * input_image[:, :, 1] + 0.1140 * input_image[:, :, 2]
        input_image = input_image.reshape([*input_image.shape, 1])

    input_image = input_image[:, :, [2, 1, 0]]
    return input_image, original_image_size


def load_image(im_path):
    success = False

    backend = os.environ.get('IMAGE_BACKEND', 'cv2')

    if backend.lower() == 'cv2':
        input_image = cv2.imread(im_path)
        if input_image is None:
            input_image = numpy.zeros((10, 10, 3), dtype='uint8')
        else:
            success = True

    elif backend.lower() == 'skimage':
        try:
            input_image = io.imread(im_path)
            success = True
        except FileNotFoundError:
            input_image = numpy.zeros((10, 10, 3), dtype='uint8')

    else:
        raise Exception('No image backend matching %s'.format(backend))

    return input_image, success


def plot_points_on_image(input_image,
                         points):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    ax.imshow(input_image)
    h, w, _ = input_image.shape

    for (cx, cy, rw, rh) in points:
        ann = patches.Ellipse((cx * w, cy * h),
                              2*rw * w,
                              2*rh * h,
                              angle=0,
                              linewidth=2,
                              fill=False,
                              zorder=2,
                              color='RED')
        ax.add_patch(ann)
    plt.show()


if __name__ == "__main__":
    filename = '/storage/git/xpl/xplplatform/data/cache/xplai-datasets-4df5d87d6e8b4cb993c3f54d14f6feb5-europe-north1/0b06dd38a37a4becbcda051248d08f9a/1/0/8/1083a299f26540b0aa80b3d04a1febb8/input.jpg'
    targets = numpy.array([30, 30, 30, 50, 50, 50, 50, 77])
    bboxes = numpy.array([[0.67770312, 0.83765625, 0.01321875, 0.00582292],
                          [0.72310156, 0.83830208, 0.01664844, 0.01069792],
                          [0.63161719, 0.83468750, 0.01277344, 0.00481250],
                          [0.33531250, 0.65038542, 0.03487500, 0.08513542],
                          [0.37752344, 0.61235417, 0.07078906, 0.14718750],
                          [0.56950000, 0.62327083, 0.08807812, 0.15160417],
                          [0.63967188, 0.62085417, 0.05428125, 0.13710417],
                          [0.89657812, 0.58006250, 0.10342187, 0.17733333]])
    input_image, _ = load_image(filename)

    plot_points_on_image(input_image.copy(), bboxes)

    for j in range(4):
        transformed_image, transform_matrix, input_size = get_random_transformed_image(input_image,
                                                                                       (512, 352),
                                                                                       j)
        print(input_size, transformed_image.shape)
        transformed_points = convert_original_cxcy_to_cropped_cxcy(transform_matrix,
                                                                   bboxes,
                                                                   targets,
                                                                   original_image_size=(input_image.shape[0],
                                                                                        input_image.shape[1]),
                                                                   cropped_image_size=(512, 352))
        plot_points_on_image(transformed_image.copy(), transformed_points)

        back_points = convert_cropped_cxcy_to_original_cxcy(transform_matrix,
                                                            transformed_points,
                                                            original_image_size=(input_image.shape[0],
                                                                                 input_image.shape[1]),
                                                            cropped_image_size=(512, 352))
        plot_points_on_image(input_image.copy(), back_points)
