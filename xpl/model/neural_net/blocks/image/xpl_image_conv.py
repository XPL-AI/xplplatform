####################################################################################################
# File: xpl_image_conv.py                                                                          #
# File Created: Monday, 12th July 2021 3:39:27 pm                                                  #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Sunday, 25th July 2021 8:22:31 pm                                                 #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from collections import OrderedDict
import torch
from typing import Optional, Union


class XPLImageConv(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 padding: Optional[list[int]] = [0, 0, 0, 0],
                 stride: Optional[int] = None,
                 bias: bool = False,
                 groups: int = 1,
                 activation: str = 'silu',
                 ):
        super().__init__()

        padding = padding if padding else [kernel_size // 2] * 4

        self.activation_class = {
            'silu': torch.nn.SiLU,
            'none': torch.nn.Identity,
        }

        self.layer = torch.nn.Sequential(OrderedDict([
            ('padd', torch.nn.ZeroPad2d(padding=padding)),
            ('conv', torch.nn.Conv2d(in_channels=input_channels,
                                     out_channels=output_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     bias=bias,
                                     groups=groups,
                                     )),
            ('batch_norm', torch.nn.BatchNorm2d(output_channels,
                                                eps=1e-3,
                                                momentum=1e-2,
                                                affine=True)),
            ('activation', self.activation_class[activation.lower()]())
        ]))

    def forward(self, x):
        return self.layer(x)
