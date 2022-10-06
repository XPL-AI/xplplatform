####################################################################################################
# File: one_vs_rest.py                                                                             #
# File Created: Thursday, 3rd May 2021 4:38:57 pm                                                  # 
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 20th October 2021 1:57:23 pm                                           #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import math
import torch

from xpl.measurer.xpl_measurer import XPLMeasurer


class OneVsRest(XPLMeasurer):

    def init_measurer(self,
                      definition: dict
                      ) -> None:
        assert 'outputs' in definition, f'{definition=} must contain "outputs"'
        self.__output_name = definition['outputs']
        assert isinstance(self.__output_name, str), f'{self.__output_name=} must be a string'

        assert 'output_is_probability' in definition, f'{definition=} must contain "output_is_probability"'
        self.__output_is_probability = definition['output_is_probability']
        assert isinstance(self.__output_is_probability, bool), f'{self.__output_is_probability=} must be a boolean'

        assert 'targets' in definition, f'{definition=} must contain "targets"'
        self.__target_name = definition['targets']
        assert isinstance(self.__target_name, str), f'{self.__target_name=} must be a string'

        assert 'num_classes' in definition, f'{definition=} must contain "targets"'
        self.__num_classes = definition['num_classes']
        assert isinstance(self.__num_classes, int), f'{self.__num_classes=} must be a int'

        self.__class_counter = torch.ones(self.__num_classes)

        if self.__output_is_probability:
            self.__loss_function = torch.nn.functional.nll_loss
            self.__probability = torch.nn.Identity()
        else:
            self.__loss_function = torch.nn.functional.cross_entropy
            self.__probability = torch.nn.Softmax(dim=1)

    def __call__(self,
                 batch: dict,
                 is_train: bool,
                 ) -> torch.Tensor:

        # TODO add asserts and check if names exist or
        # dimensions are right or the loss output is exactly
        # size batch_size
        outputs: torch.Tensor = batch[self.__output_name]
        targets: torch.Tensor = batch[self.__target_name].to(outputs.device)

        error_value = 1.0 - (outputs.argmax(dim=1) == targets).float()

        if is_train:
            for t in targets:
                # class_counter overflows after encountering the same class
                # for 2^137 times or approximately 10^38 times. I have not seen
                # anyone execute the same training procedure for more than 10^7
                # times so I don't think we are ever gonna need to check for overflow,
                # but who knows
                self.__class_counter[t] += 1

        weights = torch.sqrt(1.0 / self.__class_counter).to(outputs.device)
        weights = weights / weights.sum() * self.__num_classes
        loss_value = self.__loss_function(input=outputs,
                                          target=targets,
                                          weight=weights,
                                          reduction='none')


        # loss_value *= loss_value.detach().clone() # smaller values for less wrong samples and vice versa

        output_probabilities = self.__probability(outputs)
        entropy_value = -(output_probabilities * torch.log(output_probabilities + 1e-8)).sum(dim=1)

        # Max loss value for a random classifier must be 1.0
        loss_value /= math.log(self.__num_classes)
        entropy_value /= math.log(self.__num_classes)


        return {
            'loss': loss_value,
            'error': error_value,
            'entropy': entropy_value,
        }
