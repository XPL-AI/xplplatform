####################################################################################################
# File: xpl_sequentioal.py                                                                         #
# File Created: Sunday, 15th August 2021 2:12:05 pm                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Tuesday, 17th August 2021 5:15:06 pm                                              #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import torch

class XPLSequential(torch.nn.Sequential):

    def forward(self, input):
        for module in self:
            input = module(input)
        return input
    
    def iterative_forward(self, 
                          x: torch.Tensor
                          ) -> torch.Tensor:
        for module in self:
            if isinstance (module, XPLModule):
                input = iterative_forward(input)
        return input