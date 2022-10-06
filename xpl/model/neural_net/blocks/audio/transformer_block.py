####################################################################################################
# File: transformer_block.py                                                                       #
# File Created: Friday, 6th August 2021 11:43:52 am                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 11th August 2021 9:35:39 am                                            #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################



from typing import Optional
import torch

from xpl.model.neural_net.blocks.audio.feed_forward import FeedForward
from xpl.model.neural_net.blocks.audio.self_attention import SelfAttention


class TransformerBlock(torch.nn.Module):
    def __init__(self,
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
        self.layer_norm_first = layer_norm_first
        self.attention = SelfAttention(hidden_channels=hidden_channels,
                                       num_heads=num_heads,
                                       dropout_prob=attention_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(hidden_channels,
                                             eps=1e-5,
                                             elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.feed_forward = FeedForward(hidden_channels=hidden_channels,
                                        expand_channels=expand_channels,
                                        intermediate_dropout_prob=intermediate_dropout_prob,
                                        output_dropout_prob=output_dropout_prob
                                        )
        self.final_layer_norm = torch.nn.LayerNorm(hidden_channels,
                                                   eps=1e-5,
                                                   elementwise_affine=True)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x
