####################################################################################################
# File: pretrain_model_s.py                                                                        #
# File Created: Thursday, 15th July 2021 2:53:13 pm                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 10th November 2021 3:28:09 pm                                          #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from xpl.dataset.augment.image_tools import fix_dimension_and_normalize, load_image
from xpl.model.neural_net.backbone.image_s import ImageS
from xpl.pretrain.utils import get_state_dict, almost_equal, update_pretrained_on_server
from efficientnet_pytorch import EfficientNet


if __name__ == '__main__':

    pretrained_model = EfficientNet.from_pretrained('efficientnet-b0', advprop=True)
    pretrained_model.eval()

    xpl_model = ImageS(name='test',
                       definition={
                           'heads': ['x'],
                           'tails': ['y']
                       })
    xpl_model.eval()

    state_dict = get_state_dict(pretrained_state_dict=pretrained_model.state_dict(),
                                xpl_neural_net_state_dict=xpl_model.layers.state_dict())

    xpl_model.layers.load_state_dict(state_dict)

    test_image_path = os.path.join(os.environ['XPL_CODE_DIR'],
                                   'xpl/pretrain/test_image.jpg')

    # Preprocess image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda img: img * 2.0 - 1.0),
        torchvision.transforms.Resize((224, 224)),
    ])
    pretrained_input = transform(Image.open(test_image_path)).unsqueeze(0)

    xpl_input, success = load_image(test_image_path)
    xpl_input, _ = fix_dimension_and_normalize(xpl_input,
                                               scale_to=224,
                                               keep_aspect=True,
                                               colorspace='RGB')
    xpl_input = torch.FloatTensor(xpl_input).permute(2, 0, 1).unsqueeze(0)

    # x = xpl_input.squeeze().permute(1,2,0).numpy()
    # y = pretrained_input.squeeze().permute(1,2,0).numpy()
    # print(x.shape)
    # print(y.shape)
    # plt.subplot(2, 1, 1)
    # plt.imshow(x)
    # plt.subplot(2, 1, 2)
    # plt.imshow(y)
    # plt.show()

    # assert almost_equal(xpl_input, pretrained_input, eps=1e-1)

    # pretrained_input = 2*pretrained_input - 1

    endpoints = pretrained_model.extract_endpoints(pretrained_input)
    pretrained_output = endpoints['reduction_6']
    batch = {'x': xpl_input}
    xpl_model(batch)
    xpl_output = batch['y']

    assert almost_equal(pretrained_output, xpl_output, eps=1.2e-1)

    torch.jit.save(torch.jit.script(xpl_model), '/tmp/test.pts')
    script_model = torch.jit.load('/tmp/test.pts')
    script_model(batch)
    script_output = batch['y']

    assert almost_equal(script_output, xpl_output, eps=1e-8)

    update_pretrained_on_server(models={'image_rep': xpl_model},
                                modality='image',
                                model_size='onesize')
