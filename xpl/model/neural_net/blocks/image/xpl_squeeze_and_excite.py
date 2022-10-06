####################################################################################################
# File: xpl_squeeze_and_excite.py                                                                  #
# File Created: Monday, 19th July 2021 4:04:58 pm                                                  #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Sunday, 25th July 2021 2:28:29 pm                                                 #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from collections import OrderedDict
import torch


class XPLExciteAndSqueeze(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 squeeze_channels: int,
                 ):
        super().__init__()

        self.layers = torch.nn.Sequential(OrderedDict([
            ('avg_pool', torch.nn.AdaptiveAvgPool2d(1)),
            ('reduce', torch.nn.Conv2d(in_channels=input_channels,
                                       out_channels=squeeze_channels,
                                       kernel_size=1,
                                       bias=True)),
            ('silu', torch.nn.SiLU()),
            ('expand', torch.nn.Conv2d(in_channels=squeeze_channels,
                                       out_channels=input_channels,
                                       kernel_size=1,
                                       bias=True)),
            ('sigmoid', torch.nn.Sigmoid())
        ]))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return self.layers(x) * x
