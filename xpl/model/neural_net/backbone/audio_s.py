from typing import Dict, Optional, Union
from xpl.model.neural_net.blocks.audio.european_language_map import EuropeanLanguageMap
from xpl.model.neural_net.blocks.audio.transformer import Transformer
from xpl.model.neural_net.blocks.audio.positional_embedding import PositionalEmbedding
from xpl.model.neural_net.blocks.audio.feature_projection import FeatureProjection

from xpl.model.neural_net.blocks.audio.feature_encoder import FeatureEncoder
from nltk.tbl.feature import Feature
from requests import models
import torch
import soundfile as sf

from xpl.model.neural_net.xpl_model import XPLModel
from torchaudio.models import DeepSpeech


class AudioS(XPLModel):

    def init_neural_net(self
                        ) -> torch.nn.Module:
        return {
            'encoder': FeatureEncoder(output_channels=512),
            'projection': FeatureProjection(input_channels=512,
                                            output_channels=768,
                                            dropout_prob=0.1),
            'positional': PositionalEmbedding(hidden_channels=768,
                                              kernel_size=128,
                                              groups=16,
                                              has_layer_norm=True,
                                              dropout_prob=0.1,
                                              ),
            'transformers': Transformer(num_blocks=12,
                                        layer_drop=0.0,
                                        hidden_channels=768,
                                        num_heads=12,
                                        attention_dropout_prob=0.1,
                                        dropout_prob=0.1,
                                        expand_channels=3072,
                                        intermediate_dropout_prob=0.1,
                                        output_dropout_prob=0.1,
                                        layer_norm_first=False,
                                        ),
            'language_map': EuropeanLanguageMap(hidden_channels=768,
                                                dropout_prob=0.1,
                                                )

        }

    def forward(self,
                batch: Dict[str, torch.Tensor]
                ) -> None:

        input = batch[self.head_names[0]].to(self.device)
        encoder_output, lengths = self.encoder(input)

        projection_output, projection_norm = self.projection(encoder_output)

        mask: Optional[torch.Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = projection_output.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            projection_output[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=encoder_output.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)

        positional_output = self.positional(projection_output)

        transformer_output = self.transformers(positional_output, mask)

        output = self.language_map(transformer_output)

        batch[self.tail_names[0]] = torch.cat([transformer_output,
                                               positional_output], dim=2).permute(0, 2, 1)  # Across channels
        batch[self.tail_names[1]] = output
        if 1 > 2:
            self.decode(output)

        return None

    def decode(self,
               x: torch.Tensor
               ) -> list[list[tuple[str, int, int]]]:
        return self.language_map.decode(x)  # EuropeanLanguageMap
