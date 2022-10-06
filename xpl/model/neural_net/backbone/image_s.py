####################################################################################################
# File: image_s.py                                                                                 #
# File Created: Wednesday, 14th July 2021 10:33:10 am                                              #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Tuesday, 9th November 2021 7:17:22 pm                                             #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Dict, OrderedDict
from torch.nn.modules import padding
from xpl.model.neural_net.blocks.image.xpl_image_resnet import XPLResNetBlock
import torch
import torchvision
from xpl.model.neural_net.xpl_model import XPLModel
from xpl.model.neural_net.blocks.image.xpl_image_conv import XPLImageConv


class ImageS(XPLModel):

    def init_neural_net(self
                        ) -> torch.nn.Module:
        return {
            'layers': torch.nn.Sequential(OrderedDict([
                ('base', XPLImageConv(input_channels=3,
                                          output_channels=32,
                                          kernel_size=3,
                                          stride=2,
                                          bias=False,
                                          padding=[0, 1, 0, 1],
                                          activation='silu',)),
                ('layer_0', XPLResNetBlock(input_channels=32,
                                           output_channels=16,
                                           kernel_size=3,
                                           stride=1,
                                           expand_ratio=1,
                                           padding=[1, 1, 1, 1],
                                           has_squeeze_and_excite=True,)),
                ('layer_1', XPLResNetBlock(input_channels=16,
                                           output_channels=24,
                                           kernel_size=3,
                                           stride=2,
                                           expand_ratio=6,
                                           padding=[0, 1, 0, 1],
                                           has_squeeze_and_excite=True,)),
                ('layer_2', XPLResNetBlock(input_channels=24,
                                           output_channels=24,
                                           kernel_size=3,
                                           stride=1,
                                           expand_ratio=6,
                                           padding=[1, 1, 1, 1],
                                           has_squeeze_and_excite=True,)),
                ('layer_3', XPLResNetBlock(input_channels=24,
                                           output_channels=40,
                                           kernel_size=5,
                                           stride=2,
                                           expand_ratio=6,
                                           padding=[1, 2, 1, 2],
                                           has_squeeze_and_excite=True,)),
                ('layer_4', XPLResNetBlock(input_channels=40,
                                           output_channels=40,
                                           kernel_size=5,
                                           stride=1,
                                           expand_ratio=6,
                                           padding=[2, 2, 2, 2],
                                           has_squeeze_and_excite=True,)),
                ('layer_5', XPLResNetBlock(input_channels=40,
                                           output_channels=80,
                                           kernel_size=3,
                                           stride=2,
                                           expand_ratio=6,
                                           padding=[0, 1, 0, 1],
                                           has_squeeze_and_excite=True,)),
                ('layer_6', XPLResNetBlock(input_channels=80,
                                           output_channels=80,
                                           kernel_size=3,
                                           stride=1,
                                           expand_ratio=6,
                                           padding=[1, 1, 1, 1],
                                           has_squeeze_and_excite=True,)),
                ('layer_7', XPLResNetBlock(input_channels=80,
                                           output_channels=80,
                                           kernel_size=3,
                                           stride=1,
                                           expand_ratio=6,
                                           padding=[1, 1, 1, 1],
                                           has_squeeze_and_excite=True,)),
                ('layer_8', XPLResNetBlock(input_channels=80,
                                           output_channels=112,
                                           kernel_size=5,
                                           stride=1,
                                           expand_ratio=6,
                                           padding=[2, 2, 2, 2],
                                           has_squeeze_and_excite=True,)),
                ('layer_9', XPLResNetBlock(input_channels=112,
                                           output_channels=112,
                                           kernel_size=5,
                                           stride=1,
                                           expand_ratio=6,
                                           padding=[2, 2, 2, 2],
                                           has_squeeze_and_excite=True,)),
                ('layer_10', XPLResNetBlock(input_channels=112,
                                            output_channels=112,
                                            kernel_size=5,
                                            stride=1,
                                            expand_ratio=6,
                                            padding=[2, 2, 2, 2],
                                            has_squeeze_and_excite=True,)),
                ('layer_11', XPLResNetBlock(input_channels=112,
                                            output_channels=192,
                                            kernel_size=5,
                                            stride=2,
                                            expand_ratio=6,
                                            padding=[1, 2, 1, 2],
                                            has_squeeze_and_excite=True,)),
                ('layer_12', XPLResNetBlock(input_channels=192,
                                            output_channels=192,
                                            kernel_size=5,
                                            stride=1,
                                            expand_ratio=6,
                                            padding=[2, 2, 2, 2],
                                            has_squeeze_and_excite=True,)),
                ('layer_13', XPLResNetBlock(input_channels=192,
                                            output_channels=192,
                                            kernel_size=5,
                                            stride=1,
                                            expand_ratio=6,
                                            padding=[2, 2, 2, 2],
                                            has_squeeze_and_excite=True,)),
                ('layer_14', XPLResNetBlock(input_channels=192,
                                            output_channels=192,
                                            kernel_size=5,
                                            stride=1,
                                            expand_ratio=6,
                                            padding=[2, 2, 2, 2],
                                            has_squeeze_and_excite=True,)),
                ('layer_15', XPLResNetBlock(input_channels=192,
                                            output_channels=320,
                                            kernel_size=3,
                                            stride=1,
                                            expand_ratio=6,
                                            padding=[1, 1, 1, 1],
                                            has_squeeze_and_excite=True,)),
                ('final', XPLImageConv(input_channels=320,
                                       output_channels=1280,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       padding=[0, 0, 0, 0],
                                       activation='silu',)),
            ]))
        }

    def forward(self,
                batch: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        input = batch[self.head_names[0]].to(self.device)

        output = self.layers(2.0 * input - 1.0)
        batch[self.tail_names[0]] = output
        return batch
