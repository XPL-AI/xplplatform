####################################################################################################
# File: positional_embedding.py                                                                    #
# File Created: Friday, 6th August 2021 11:53:40 am                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 11th August 2021 9:35:22 am                                            #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


import torch


class PositionalEmbedding(torch.nn.Module):

    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int,
                 groups: int,
                 has_layer_norm: bool,
                 dropout_prob: float,
                 ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.has_layer_norm = has_layer_norm
        self.conv = torch.nn.Conv1d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2,
                                    groups=groups,
                                    )
        self.conv = torch.nn.utils.weight_norm(self.conv,
                                               name="weight",
                                               dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0
        if self.has_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(hidden_channels,
                                                 eps=1e-5,
                                                 elementwise_affine=True)
        self.drop_out = torch.nn.Dropout(p=dropout_prob)

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            if hook.__module__ == 'torch.nn.utils.weight_norm' and \
                    hook.__class__.__name__ == 'WeightNorm':
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        residual = x
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)

        x = x + residual

        if self.has_layer_norm:
            x = self.layer_norm(x)
        x = self.drop_out(x)

        return x
