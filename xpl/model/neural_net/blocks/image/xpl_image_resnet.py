####################################################################################################
# File: xpl_image_resnet.py                                                                        #
# File Created: Monday, 19th July 2021 3:55:31 pm                                                  #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Sunday, 25th July 2021 8:48:39 pm                                                 #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import OrderedDict
from torch.nn.modules.normalization import GroupNorm
from xpl.model.neural_net.blocks.image.xpl_squeeze_and_excite import XPLExciteAndSqueeze
import torch
from xpl.model.neural_net.blocks.image.xpl_image_conv import XPLImageConv


class XPLResNetBlock(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 expand_ratio: int = 1,
                 padding: list[int] = [1, 1, 1, 1],
                 has_squeeze_and_excite: bool = True,
                 ):
        super().__init__()

        mid_channels = input_channels * expand_ratio

        expand_layer = torch.nn.Identity()
        excite_layer = torch.nn.Identity()

        if expand_ratio > 1:
            expand_layer = XPLImageConv(input_channels=input_channels,
                                        output_channels=mid_channels,
                                        kernel_size=1,
                                        stride=1,
                                        groups=1,
                                        bias=False,
                                        )

        depthwise_layer = XPLImageConv(input_channels=mid_channels,
                                       output_channels=mid_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       groups=mid_channels,
                                       bias=False,
                                       padding=padding
                                       )
        if has_squeeze_and_excite:
            excite_layer = XPLExciteAndSqueeze(input_channels=mid_channels,
                                               squeeze_channels=input_channels // 4)

        project_layer = XPLImageConv(input_channels=mid_channels,
                                     output_channels=output_channels,
                                     kernel_size=1,
                                     stride=1,
                                     groups=1,
                                     bias=False,
                                     activation='none',
                                     )
        self.layer = torch.nn.Sequential(OrderedDict([
            ('expand', expand_layer),
            ('depthwise', depthwise_layer),
            ('excite', excite_layer),
            ('project', project_layer)
        ]))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        y = self.layer(x)
        if y.size() == x.size():
            y += x
        return y
