####################################################################################################
# File: feed_forward.py                                                                            #
# File Created: Tuesday, 10th August 2021 5:28:58 pm                                               #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Tuesday, 17th August 2021 11:23:47 am                                             #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


import torch


class FeedForward(torch.nn.Module):

    def __init__(
            self,
            hidden_channels: int,
            expand_channels: int,
            intermediate_dropout_prob: float,
            output_dropout_prob: float,
    ):
        super().__init__()
        self.intermediate_dense = torch.nn.Linear(hidden_channels, expand_channels)
        self.intermediate_dropout = torch.nn.Dropout(intermediate_dropout_prob)
        self.output_dense = torch.nn.Linear(expand_channels, hidden_channels)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)
        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
