####################################################################################################
# File: utils.py                                                                                   #
# File Created: Wednesday, 7th July 2021 1:45:35 pm                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Tuesday, 28th September 2021 7:02:26 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import torch
from torch.jit import Error
import torchvision
import torchaudio
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import urllib
import random
import torch.nn.functional as F


def load_image_from_disk(absolute_path: str,
                         image_size: tuple[int, int],
                         ) -> torch.Tensor:
    try:
        image = torchvision.io.read_image(path=absolute_path,
                                          mode=torchvision.io.image.ImageReadMode.RGB)
    except RuntimeError as e:
        image = Image.open(absolute_path)
        image = image.seek(0)
        if image is None:
            return None
        else:
            image = torchvision.transforms.ToTensor()(image)

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.dtype == torch.float32:
        assert image.max() <= 1. and image.min() >= 0.0, f'{image=} values must be in [0,1)'
        pass

    elif image.dtype == torch.uint8:
        image = image.float() / 255.0

    elif image.dtype == torch.int16:
        image = image.float() / 65535.0

    else:
        raise BaseException(f'Unknown image type {image.type()} for {absolute_path=}')

    # at this point, image must be in the shape of [C, W, H]
    # and values must be between [0, 1]
    image = image * 2 - 1
    # Now the value distribution is shifted to approximately N~(mu=0, sigma=1)
    # So we don't need to normalize the image itself
    print(image_size)
    
    if image.shape[1] > image.shape[2]:
        return F.interpolate(image.unsqueeze(0),
                             size=image_size,
                             mode='bilinear').squeeze(0)
    else:
        return F.interpolate(image.unsqueeze(0),
                             size=(image_size[1], image_size[0]),
                             mode='bilinear').squeeze(0)


def load_audio_from_disk(absolute_path: str,
                         default_sample_rate: int = 16000
                         ) -> torch.Tensor:
    try:
        audio, sample_rate = torchaudio.backend.sox_io_backend.load(filepath=absolute_path,
                                                                    normalize=True,  # we need to normalize it ourselves
                                                                    channels_first=True)  # This must be here
        audio = audio.mean(axis=0).unsqueeze(0)  # Getting rid of multiple channels for sterio audio!
        # TODO make sure audio values are float and in the range of [-1, 1]!

    except RuntimeError as runtime_error:
        print(f'problem loading {absolute_path=}')
        raise runtime_error
    if not sample_rate == default_sample_rate:
        # TODO make sure resample gives the best quality
        audio = torchaudio.functional.resample(waveform=audio,
                                               orig_freq=sample_rate,
                                               new_freq=default_sample_rate)

    # Now we normalize to : ~N(mu=0, var=1)
    audio = (audio - audio.mean()) / (audio.var() + 1e-5).sqrt()
    return audio


def load_and_augment_image_from_disk(absolute_path: str
                                     ) -> torch.Tensor:
    try:
        image = torchvision.io.read_image(path=absolute_path,
                                          mode=torchvision.io.image.ImageReadMode.RGB)
    except RuntimeError as e:
        image = Image.open(absolute_path)
        image = image.seek(0)
        if image is None:
            image = torch.rand((3, 60, 60))
        else:
            image = torchvision.transforms.ToTensor()(image)

    image, bounding_box = augmenting_the_image(torchvision.transforms.ToPILImage()(image))

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.dtype == torch.float32:
        assert image.max() <= 1. and image.min() >= 0.0, f'{image=} values must be in [0,1)'
        pass

    elif image.dtype == torch.uint8:
        image = image.float() / 255.0

    elif image.dtype == torch.int16:
        image = image.float() / 65535.0

    else:
        raise BaseException(f'Unknown image type {image.type()} for {absolute_path=}')

    # at this point, image must be in the shape of [C, W, H]
    # and values must be between [0, 1]
    # Now the value distribution is shifted to approximately N~(mu=0, sigma=1)
    # So we don't need to normalize the image itself
    image = image * 2 - 1
    return image, bounding_box


def augmenting_the_image(original_image):
    # getting access to the web page
    page = requests.get("https://picsum.photos/")
    soup = bs(page.text, 'html.parser')
    for pic in soup.findAll('div', class_="content-section-dark"):
        for background in pic.findAll('div', class_="container mx-auto flex flex-wrap"):
            img = Image.open(urllib.request.urlopen(background.get_text().split()[23]))
    # getting the right starting point so we don't go out of bounds
    x = img.size[0] - original_image.size[0]
    y = img.size[1] - original_image.size[1]
    x = random.randint(0, x)
    y = random.randint(0, y)
    # pasting the sign onto the backgroud
    img.paste(original_image, (x, y))
    return torchvision.transforms.ToTensor()(img), [x, y, x+original_image.size[0], y + original_image.size[1]]
