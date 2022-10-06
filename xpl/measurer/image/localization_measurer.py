####################################################################################################
# File: yolox_measurer.py                                                                          #
# File Created: Wednesday, 20th October 2021 1:55:35 pm                                            #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Thursday, 11th November 2021 12:15:33 pm                                          #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import math
from numpy import int64
import torch
import traceback

from xpl.measurer.xpl_measurer import XPLMeasurer


class LocalizationMeasurer(XPLMeasurer):

    def init_measurer(self,
                      definition: dict
                      ) -> None:
        assert 'concept_group_name' in definition, f'{definition=} must contain "outputs"'
        self.__concept_name = definition['concept_group_name']

        assert 'num_classes' in definition, f'{definition=} must contain "targets"'
        self.__num_classes = definition['num_classes']
        assert isinstance(self.__num_classes, int), f'{self.__num_classes=} must be a int'

        self.__class_counter = torch.ones(self.__num_classes)
        self.__objectiveness_counter = torch.ones(2)

    def __call__(self,
                 batch: dict[torch.Tensor],
                 is_train: bool,
                 ) -> torch.Tensor:

        pred_objectiveness = batch[f'pred_{self.__concept_name[0]}_objectiveness']
        pred_location = batch[f'pred_{self.__concept_name[0]}_locations']
        pred_class = batch[f'pred_{self.__concept_name[0]}_label']
        pred_mask = batch[f'pred_{self.__concept_name[0]}_mask']

        target_objectiveness = batch[f'{self.__concept_name[0]}_objectiveness'].to(pred_objectiveness.device)
        target_location = batch[f'{self.__concept_name[0]}_locations'].to(pred_location.device)
        target_class = batch[f'{self.__concept_name[0]}_label'].to(pred_class.device)

        target_location = target_location[pred_mask, :, :]
        target_class = target_class[pred_mask, :]

        (target_objectiveness,
         target_location,
         target_class,
         pred_objectiveness,
         pred_location,
         pred_class) = self.align_targets_with_predictions(target_objectiveness=target_objectiveness,
                                                           target_location=target_location,
                                                           target_class=target_class,
                                                           pred_objectiveness=pred_objectiveness,
                                                           pred_location=pred_location,
                                                           pred_class=pred_class,
                                                           pred_mask=pred_mask)

        target_objectiveness[target_objectiveness > 1] = 1

        (objectiveness_loss,
         true_positive_rate,
         true_negative_rate,
         false_positive_rate,
         false_negative_rate) = self.objectiveness_loss(pred_objectiveness=pred_objectiveness,
                                                        target_objectiveness=target_objectiveness,
                                                        is_train=is_train)

        classification_loss, classification_error = self.classify(pred_class=pred_class,
                                                                  target_class=target_class,
                                                                  is_train=is_train)

        return {
            'loss':  classification_loss.mean() + objectiveness_loss.mean(),
            'objectiveness_loss': objectiveness_loss.mean().item(),
            'classification_loss': classification_loss.mean().item(),
            'classification_error': classification_error.mean().item(),
            'tpr': true_positive_rate,
            'fpr': false_positive_rate,
            'tnr': true_negative_rate,
            'fnr': false_negative_rate,
        }

    def align_targets_with_predictions(self,
                                       target_objectiveness,
                                       target_location,
                                       target_class,
                                       pred_objectiveness,
                                       pred_location,
                                       pred_class,
                                       pred_mask):

        seq_length = target_class.shape[1]
        pred_location = pred_location[:, :seq_length, :]
        pred_class = pred_class[:, :seq_length, :].reshape(-1, pred_class.shape[-1])
        target_class = target_class.reshape(-1)

        return (target_objectiveness,
                target_location,
                target_class,
                pred_objectiveness,
                pred_location,
                pred_class)

    def classify(self,
                 pred_class,
                 target_class,
                 is_train):
        if is_train:
            for t in target_class:
                # class_counter overflows after encountering the same class
                # for 2^137 times or approximately 10^38 times. I have not seen
                # anyone execute the same training procedure for more than 10^7
                # times so I don't think we are ever gonna need to check for overflow,
                # but who knows
                self.__class_counter[t] += 1

        weights = (1.0 / self.__class_counter).to(pred_class.device)
        weights = weights / weights.sum() * self.__num_classes * 100
        loss_value = torch.nn.functional.cross_entropy(input=pred_class,
                                                       target=target_class,
                                                       weight=weights,
                                                       reduction='none')
        error_value = 1.0 - (pred_class.argmax(dim=1) == target_class).float()
        error_value = error_value[target_class > 0]
        return loss_value, error_value

    def objectiveness_loss(self,
                           pred_objectiveness,
                           target_objectiveness,
                           is_train):
        if is_train:
            self.__objectiveness_counter[0] += (target_objectiveness == 0).float().sum().item()
            self.__objectiveness_counter[1] += (target_objectiveness > 0).float().sum().item()

        weights = (1.0 / self.__objectiveness_counter).to(pred_objectiveness.device)
        weights = weights / weights.sum() * 2
        print(weights, target_objectiveness.max(), target_objectiveness.min(), pred_objectiveness.max(), pred_objectiveness.min())

        loss_value = -weights[1] * target_objectiveness * torch.log(pred_objectiveness+1e-8) + \
                     -weights[0] * (1.0-target_objectiveness) * (torch.log(1.0 - pred_objectiveness + 1e-8))

        true_positive_rate = ((pred_objectiveness[target_objectiveness > 0] > 0.5).float() == 1).float().mean().item()
        true_negative_rate = ((pred_objectiveness[target_objectiveness == 0] > 0.5).float() == 0).float().mean().item()
        false_positive_rate = ((pred_objectiveness[target_objectiveness == 0] > 0.5).float() == 1).float().mean().item()
        false_negative_rate = ((pred_objectiveness[target_objectiveness > 0] > 0.5).float() == 0).float().mean().item()

        return loss_value, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate
