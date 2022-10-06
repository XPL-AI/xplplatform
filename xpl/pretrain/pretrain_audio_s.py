####################################################################################################
# File: pretrain_audio_s.py                                                                        #
# File Created: Tuesday, 3rd August 2021 1:45:40 pm                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 15th September 2021 4:53:20 pm                                         #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import os
from typing import OrderedDict
import torch
import transformers
import soundfile
from transformers import Wav2Vec2ForCTC

from xpl.model.neural_net.backbone.audio_s import AudioS
from xpl.dataset.augment.utils import load_audio_from_disk
from xpl.pretrain.utils import get_state_dict, almost_equal, update_pretrained_on_server


if __name__ == '__main__':

    # Model definition
    pretrained_model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
    pretrained_model.eval()
    print(pretrained_model)
    xpl_model = AudioS(name='test',
                       definition={
                           'heads': ['x'],
                           'tails': ['y', 'text']
                       })
    xpl_model.eval()
    print(xpl_model)

    vocab_inv = {'_': 0, '<': 1, '>': 2, '?': 3, ' ': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12,
             'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26,
             '\'': 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31}

    pretrained_vocab = {v:k for k, v in vocab_inv.items()}
    xpl_alphabet = xpl_model.language_map.alphabet
    
    pretrain_to_xpl_map = [xpl_alphabet.find(pretrained_vocab[i].lower()) for i in range(len(pretrained_vocab))]
    pretrain_to_xpl_map[0] = 0
    pretrained_weight = pretrained_model.lm_head.state_dict()['weight']
    pretrained_bias = pretrained_model.lm_head.state_dict()['bias']
    xpl_weight = xpl_model.language_map.state_dict()['layer.1.weight']
    xpl_bias = xpl_model.language_map.state_dict()['layer.1.bias']
    xpl_bias = -10 * torch.ones_like(xpl_bias)

    print(pretrained_weight.shape)
    print(pretrained_bias.shape)
    xpl_weight[pretrain_to_xpl_map, :] = pretrained_weight
    xpl_bias[pretrain_to_xpl_map] = pretrained_bias
    lang_state_dict = OrderedDict({'layer.1.weight': xpl_weight,
                                  'layer.1.bias': xpl_bias})
    

    xpl_model.language_map.load_state_dict(lang_state_dict)

    # Load pretrained models
    encoder_state_dict = get_state_dict(pretrained_model.wav2vec2.feature_extractor.state_dict(),
                                        xpl_model.encoder.state_dict())
    xpl_model.encoder.load_state_dict(encoder_state_dict)

    projection_state_dict = get_state_dict(pretrained_model.wav2vec2.feature_projection.state_dict(),
                                           xpl_model.projection.state_dict())
    xpl_model.projection.load_state_dict(projection_state_dict)

    positional_state_dict = get_state_dict(torch.nn.Sequential(pretrained_model.wav2vec2.encoder.pos_conv_embed,
                                                               pretrained_model.wav2vec2.encoder.layer_norm).state_dict(),
                                           xpl_model.positional.state_dict()
                                           )
    xpl_model.positional.load_state_dict(positional_state_dict)

    transformer_state_dict = get_state_dict(pretrained_model.wav2vec2.encoder.layers.state_dict(),
                                            xpl_model.transformers.state_dict()
                                            )
    xpl_model.transformers.load_state_dict(transformer_state_dict)


    # Check inputs
    audio_path = os.path.join(os.environ['XPL_CODE_DIR'],
                              'xpl/pretrain/test_audio.flac')
    pretrained_input, _ = soundfile.read(audio_path)
    pretrained_input = torch.Tensor(pretrained_input).unsqueeze(0)
    pretrained_input = (pretrained_input - pretrained_input.mean()) / (pretrained_input.var() + 1e-5).sqrt()
    pretrained_input = torch.cat([pretrained_input, pretrained_input, pretrained_input])

    xpl_input = load_audio_from_disk(audio_path)
    xpl_input = torch.cat([xpl_input, xpl_input, xpl_input])
    
    print('testing inputs:')
    assert almost_equal(pretrained_input, xpl_input)
    print(pretrained_input.shape)
    print(xpl_input.shape)

    # Check encoders
    pretrained_encoder_output = pretrained_model.wav2vec2.feature_extractor(pretrained_input)
    input_length = torch.FloatTensor([[xpl_input.shape[-1], xpl_input.shape[-1], xpl_input.shape[-1]]])
    xpl_encoder_output, encoder_output_length = xpl_model.encoder(xpl_input.unsqueeze(1), input_length)
    print(encoder_output_length, xpl_encoder_output.shape)
    print('testing encoder:')
    assert almost_equal(xpl_encoder_output, pretrained_encoder_output)

    # Check projections
    xpl_projection_output, _ = xpl_model.projection(xpl_encoder_output)
    pretrained_projection_output, _ = pretrained_model.wav2vec2.feature_projection(pretrained_encoder_output.transpose(1, 2))
    print('testing projection:')
    assert almost_equal(xpl_projection_output, pretrained_projection_output)

    pretrained_output = pretrained_model(pretrained_input).logits
    print(pretrained_output.shape)
    batch = {'x': xpl_input.unsqueeze(1)}
    xpl_model(batch)
    xpl_output = batch['y']
    xpl_text = batch['text']

    print('testing the whole thing:')
    #assert almost_equal(xpl_output, pretrained_output, eps=0.13)

    torch.jit.save(torch.jit.script(xpl_model), '/tmp/test.pts')
    script_model = torch.jit.load('/tmp/test.pts')
    script_model(batch)
    script_output = batch['y']
    script_text = batch['text']
    print(script_model.decode(script_text))

    assert almost_equal(script_output, xpl_output, eps=1e-8)
    
    update_pretrained_on_server(models={'audio_rep': xpl_model},
                                modality='audio',
                                model_size='onesize')
