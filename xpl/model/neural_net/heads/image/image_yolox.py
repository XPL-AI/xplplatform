####################################################################################################
# File: image_yolox.py                                                                             #
# File Created: Monday, 30th August 2021 2:25:25 pm                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Thursday, 11th November 2021 1:28:39 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Dict
from xpl.model.neural_net.blocks.image.xpl_gru import XPLGru
import numpy
import torch

from xpl.model.neural_net.xpl_model import XPLModel


class YoloX(XPLModel):

    def init_neural_net(self
                        ) -> Dict[str, torch.nn.Module]:
        input_channels = self.definition['input_channels']['target_channels']
        output_channels = self.definition['output_channels']
        return {
            'classification_rep': torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channels,
                                out_channels=input_channels,
                                kernel_size=7,
                                padding=3,
                                stride=1,
                                groups=input_channels//32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
            ),
            'location_rep': torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channels,
                                out_channels=input_channels,
                                kernel_size=7,
                                padding=3,
                                stride=1,
                                groups=input_channels//32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
            ),
            'objectiveness_layer': torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channels,
                                out_channels=1,
                                kernel_size=7,
                                padding=3,
                                stride=1,
                                groups=1),
                torch.nn.Sigmoid()
            ),
            'classification_layer': XPLGru(in_channels=input_channels,
                                           out_channels=output_channels),

            'location_layer': XPLGru(in_channels=input_channels,
                                     out_channels=4),

        }

    def forward(self,
                batch: dict[str, torch.Tensor]
                ) -> dict[str, torch.Tensor]:

        input = batch[self.head_names[0]].to(self.device)
        max_try = 1
        if f'{self.tail_names[0]}_label' in batch.keys():
            max_try = batch[f'{self.tail_names[0]}_label'].shape[3]

        location_rep = self.location_rep(input)
        pred_objectiveness = self.objectiveness_layer(location_rep).squeeze(1)

        mask = pred_objectiveness > 0.45
        if self.training and f'{self.tail_names[0]}_objectiveness' in batch:
            mask = torch.logical_or(mask, batch[f'{self.tail_names[0]}_objectiveness'].to(mask.device) > 0)
        
        class_rep = self.classification_rep(input)

        batch_size, _, h, w = location_rep.shape

        # bringing the channel to the last dimension
        location_rep = location_rep.permute(0, 2, 3, 1)
        class_rep = class_rep.permute(0, 2, 3, 1)

        # threating each output seperately
        location_rep = location_rep[mask, :].unsqueeze(0)
        class_rep = class_rep[mask, :].unsqueeze(0)

        # TODO: feed objectiveness to the classification layer
        pred_label = self.classification_layer(class_rep, max_try=max_try)

        # TODO: feed the predicted classes to the location layer
        pred_location = self.location_layer(location_rep, max_try=max_try)

        pred_label = pred_label.permute(1, 0, 2)
        pred_location = pred_location.permute(1, 0, 2)

        batch[f'pred_{self.tail_names[0]}_mask'] = mask
        batch[f'pred_{self.tail_names[0]}_label'] = pred_label
        batch[f'pred_{self.tail_names[0]}_objectiveness'] = pred_objectiveness
        batch[f'pred_{self.tail_names[0]}_locations'] = pred_location

        return batch
