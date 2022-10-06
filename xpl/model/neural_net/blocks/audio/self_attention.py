####################################################################################################
# File: self_attention.py                                                                          #
# File Created: Tuesday, 10th August 2021 5:22:31 pm                                               #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 11th August 2021 9:36:41 am                                            #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

from typing import Optional
import torch


class SelfAttention(torch.nn.Module):

    def __init__(
            self,
            hidden_channels: int,
            num_heads: int,
            dropout_prob: float = 0.0,
    ):
        super().__init__()
        head_dim = hidden_channels // num_heads
        if head_dim * num_heads != hidden_channels:
            raise ValueError(f"`hidden_channels ({hidden_channels})` is not divisible by `num_heads ({num_heads})`")

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.v_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.q_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.out_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=True)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        if x.ndim != 3 or x.shape[2] != self.hidden_channels:
            raise ValueError(
                f"The expected input shape is (batch, sequence, hidden_channels=={self.hidden_channels}). "
                f"Found {x.shape}."
            )
        batch_size, length, hidden_channels = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        weights = self.scaling * (q @ k)  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights,
                                              dim=-1)
        weights = torch.nn.functional.dropout(weights,
                                              p=self.dropout_prob,
                                              training=self.training)

        output = weights @ v  # B, nH, L, Hd
        output = output.transpose(2, 1).reshape(batch_size, length, hidden_channels)

        output = self.out_proj(output)
        return output
