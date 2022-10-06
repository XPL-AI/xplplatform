####################################################################################################
# File: feature_encoder.py                                                                         #
# File Created: Wednesday, 4th August 2021 6:11:15 pm                                              #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 11th August 2021 9:34:30 am                                            #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


import math
from typing import Optional, OrderedDict, Tuple
import torch


class FeatureEncoder(torch.nn.Module):

    def __init__(self,
                 output_channels: int = 512,
                 kernel_sizes: Optional[list[int]] = None,
                 ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes if kernel_sizes else [10, 3, 3, 3, 3, 2, 2]

        input_channels = 1
        conv_layers = OrderedDict()
        for i, kernel_size in enumerate(self.kernel_sizes):
            conv_layers[f'conv_{i}'] = torch.nn.Conv1d(in_channels=input_channels,
                                                       out_channels=output_channels,
                                                       kernel_size=kernel_size,
                                                       stride=math.ceil(kernel_size/3+1),
                                                       bias=False)
            if input_channels == 1:
                conv_layers['group_norm'] = torch.nn.GroupNorm(num_groups=output_channels,
                                                               num_channels=output_channels,
                                                               eps=1e-5,
                                                               affine=True)
                input_channels = output_channels
            conv_layers[f'glue_{i}'] = torch.nn.GELU()

        self.layer = torch.nn.Sequential(conv_layers)

    def forward(self,
                x: torch.Tensor,
                length: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if length is not None:
            for kernel_size in self.kernel_sizes:
                length = torch.div(length - kernel_size, math.ceil(kernel_size/3+1), rounding_mode='floor') + 1

        return self.layer(x), length
