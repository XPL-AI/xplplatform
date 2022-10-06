####################################################################################################
# File: feature_projection.py                                                                      #
# File Created: Wednesday, 4th August 2021 6:04:27 pm                                              #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Friday, 6th August 2021 2:55:23 pm                                                #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Tuple
import torch


class FeatureProjection(torch.nn.Module):

    def __init__(self,
                 input_channels: int = 512,
                 output_channels: int = 768,
                 dropout_prob: float = 0.1
                 ) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=input_channels,
                                             eps=1e-5,
                                             elementwise_affine=True)
        self.projection = torch.nn.Linear(in_features=input_channels,
                                          out_features=output_channels,
                                          bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_prob, inplace=False)

    def forward(self,
                x: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = x.transpose(1, 2)
        x_norm = self.layer_norm(x)
        y = self.projection(x_norm)
        y = self.dropout(y)

        return y, x_norm
