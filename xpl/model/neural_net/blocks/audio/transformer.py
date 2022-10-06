####################################################################################################
# File: transformer.py                                                                             #
# File Created: Tuesday, 10th August 2021 5:46:51 pm                                               #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Monday, 16th August 2021 8:51:18 am                                               #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


from typing import Optional
from xpl.model.neural_net.blocks.audio.transformer_block import TransformerBlock
import torch


class Transformer(torch.nn.Module):
    def __init__(self,
                 num_blocks: int,
                 layer_drop: float,
                 hidden_channels: int,
                 num_heads: int,
                 attention_dropout_prob: float,
                 expand_channels: int,
                 dropout_prob: float,
                 intermediate_dropout_prob: float,
                 output_dropout_prob: float,
                 layer_norm_first: bool
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.layer_drop = layer_drop
        #self.layers: list[TransformerBlock] = []
        for i in range(num_blocks):
            layer = TransformerBlock(hidden_channels=hidden_channels,
                                     num_heads=num_heads,
                                     attention_dropout_prob=attention_dropout_prob,
                                     expand_channels=expand_channels,
                                     dropout_prob=dropout_prob,
                                     intermediate_dropout_prob=intermediate_dropout_prob,
                                     output_dropout_prob=output_dropout_prob,
                                     layer_norm_first=layer_norm_first
                                     )
            self.add_module(f'transformer_{i}', layer)
            #self.layers.append(layer)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        for layer in self.children():
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x = layer(x, attention_mask)
        return x
