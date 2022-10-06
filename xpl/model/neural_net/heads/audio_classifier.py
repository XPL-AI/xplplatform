####################################################################################################
# File: audio_classifier.py                                                                        #
# File Created: Wednesday, 14th July 2021 2:10:31 pm                                               #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 14th July 2021 3:52:56 pm                                              #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import torch

from xpl.model.neural_net.xpl_model import XPLModel


class AudioClassifier(XPLModel):

    def init_neural_net(self
                        ) -> torch.nn.Module:
        return torch.nn.Conv1d(in_channels=self.definition['input_channels'],
                               out_channels=self.definition['output_channels'],
                               kernel_size=1)

    def forward(self,
                batch: dict
                ) -> None:
        input = batch[self.head_names[0]].to(self.device)
        output = self.neural_net(input).mean(-1)
        batch[self.tail_names[0]] = output
        return None
