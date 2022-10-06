####################################################################################################
# File: image_xs.py                                                                                #
# File Created: Wednesday, 14th July 2021 10:32:46 am                                              #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 14th July 2021 2:08:27 pm                                              #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import torch

from xpl.model.neural_net.xpl_model import XPLModel


class ImageXS(XPLModel):

    def init_neural_net(self
                        ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=5,
                            stride=2,
                            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=512,
                            kernel_size=5,
                            stride=2),
            torch.nn.ReLU()
        )

    def forward(self,
                batch: dict
                ) -> None:
        input = batch[self.head_names[0]].to(self.device)
        output = self.neural_net(input)
        batch[self.tail_names[0]] = output
        return None
