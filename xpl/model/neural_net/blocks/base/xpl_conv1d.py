####################################################################################################
# File: xpl_conv1d.py                                                                              #
# File Created: Sunday, 15th August 2021 2:22:43 pm                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Sunday, 15th August 2021 2:23:43 pm                                               #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import collections
import numpy as np

import torch


class UndeadConv1d(torch.nn.Module):

    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size,
                 padding,
                 has_norm_layer=True,
                 dropout=0.2):
        super(UndeadConv1d, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding

        self.layer_dict = collections.OrderedDict({})
        if has_norm_layer:
            self.layer_dict['norm'] = torch.nn.GroupNorm(input_c//2, input_c)
        self.layer_dict['conv'] = torch.nn.Conv1d(input_c, output_c, kernel_size, padding=padding, groups=1)
        self.layer_dict['dropout'] = torch.nn.Dropout(dropout)

        self.layers = torch.nn.Sequential(self.layer_dict)

        self.stats = torch.zeros(output_c)
        self.active_stats = torch.zeros(output_c)
        self.passive_stats = torch.zeros(output_c)
        self.iterative_input = None

    def forward(self, x, add_noise=False):

        if add_noise and self.training and np.random.rand() > .8:
            noise = torch.randn(x.shape) * np.random.rand() * x.std().item() * 0.01
            noise[x == 0] = 0
            x = x + torch.nn.functional.relu(noise).to(x.device)
        y = self.layers(x) 
        
        if y.shape[1] == x.shape[1]:
            y = F.relu(y + x)
        else:
            y = F.relu(y)

        if self.training:
            if self.stats.device != y.device:
                self.stats.to(y.device)
                self.active_stats.to(y.device)
                self.passive_stats.to(y.device)

            self.passive_stats += 1
            # self.active_stats += 1
            current_stat = (y.detach().mean(2).mean(0))
            self.passive_stats[current_stat > 0] = 0
            # self.active_stats[current_stat == 0] = 0
            self.stats = self.passive_stats  # + self.active_stats

        return y

    def forward_residual(self, x, add_noise=False):
        y = self.layers(x)
        margin = self.kernel_size // 2 - self.padding
        if margin == 0:
            return y + x
        else:
            return y + x[:, :, margin:-margin]

    def forward_iterative(self, x):
        if x is None:
            return None

        if self.iterative_input is None:
            self.iterative_input = x
        else:
            self.iterative_input = torch.cat(
                [self.iterative_input, x], dim=2)

        if self.iterative_input.shape[2] < self.kernel_size:
            return None

        self.iterative_input = self.iterative_input[:, :, -self.kernel_size:]

        x = self.layers(self.iterative_input)
        x = x[:, :, self.padding].unsqueeze(2)
        return x

    def forward_residual_iterative(self, x):
        if x is None:
            return None

        if self.iterative_input is None:
            self.iterative_input = x
        else:
            self.iterative_input = torch.cat(
                [self.iterative_input, x], dim=2)

        if self.iterative_input.shape[2] < self.kernel_size:
            return None

        self.iterative_input = self.iterative_input[:, :, -self.kernel_size:]

        residual = self.iterative_input[:, :,
                                        self.kernel_size // 2].unsqueeze(2)
        x = self.layers(self.iterative_input) + residual
        x = x[:, :, self.padding].unsqueeze(2)
        return x

    def update_stats(self):
        mask = self.stats > 100
        if mask.sum() > 0:
            weights = list(self.layer_dict['conv'].parameters())[0]
            bias = list(self.layer_dict['conv'].parameters())[1]
            new_weights = weights.clone()
            new_bias = bias.clone()
            new_weights[mask, :, :] = 0.001 * torch.randn((mask.sum(),
                                                           weights.shape[1],
                                                           weights.shape[2])).to(weights.device)
            new_bias[mask] = 0.001 * torch.randn(mask.sum()).to(bias.device)
            weights.data.copy_(new_weights)
            bias.data.copy_(new_bias)

            self.stats[mask] = 0
