####################################################################################################
# File: xpl_gru.py                                                                                 #
# File Created: Tuesday, 2nd November 2021 8:06:58 pm                                              #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Thursday, 11th November 2021 1:25:11 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Optional
import torch


class XPLGru(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XPLGru, self).__init__()

        self.gru = torch.nn.GRU(in_channels,
                                in_channels)
        self.out = torch.nn.Linear(in_channels,
                                   out_channels)
        self.max_try: int = 1

    def forward(self,
                hidden_rep: torch.Tensor,
                max_try: int,
                additional_input: Optional[torch.Tensor] = None):
        if self.max_try < max_try:
            self.max_try = max_try

        input = torch.zeros_like(hidden_rep)

        outputs: list[torch.Tensor] = []
        for i in range(self.max_try):
            input, hidden_rep = self.gru(input, hidden_rep)
            output = self.out(torch.nn.functional.relu(input))
            outputs.append(output)

        return torch.cat(outputs)
