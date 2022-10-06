####################################################################################################
# File: language_map.py                                                                            #
# File Created: Tuesday, 17th August 2021 11:17:56 am                                              #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Monday, 23rd August 2021 10:28:23 am                                              #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################


import torch


class EuropeanLanguageMap(torch.nn.Module):

    def __init__(
            self,
            hidden_channels: int,
            dropout_prob: float,
    ):
        super().__init__()

        self.punctuations = u'_ <>?\'"!'
        self.english_chars = u'abcdefghijklmnopqrstuvwxyz'
        self.latin_extra_letters = u'ÐßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿāăąćčďđēĕėęğĢģīįİıķļłńņňŋőœśşšţťŧūűųźżž'
        self.greek_letter = u'αβγδεζηθικλμνξοπρστυφχψω'
        self.cyrillic_letters = u'абвгдежзийклмнопрстуфхцчшщъыьэюяёђѓєїљњћќўџґ'

        self.alphabet: str = u''.join([self.punctuations,
                                      self.english_chars,
                                      self.latin_extra_letters,
                                      self.greek_letter,
                                      self.cyrillic_letters])

        self.vocab = {i+1: c for i, c in enumerate(self.alphabet)}

        self.layer = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_prob, inplace=False),
            torch.nn.Linear(in_features=hidden_channels,
                            out_features=len(self.alphabet),
                            bias=True)
        )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.layer(x)
        return x

    def decode(self, # EuropeanLanguageMap
               x: torch.Tensor
               ) -> list[list[tuple[str, int, int]]]:
        batch_size, sequence_length, _ = x.shape
        output = x.argmax(-1)

        texts: list[list[tuple[str, int, int]]] = []
        for b in range(batch_size):
            words: list[tuple[str, int, int]] = []
            unprocessed_text = ''
            for i in range(sequence_length):
                unprocessed_text += self.alphabet[output[b][i].item()]
            current_word = ''
            current_word_start = -1
            current_word_end = -1
            for i, c in enumerate(unprocessed_text):
                if c == self.punctuations[0]:
                    continue
                elif c in self.punctuations:
                    current_word_end = i
                    if len(current_word) > 0:
                        words.append((current_word, current_word_start, current_word_end))
                    current_word = ''
                    current_word_start = -1
                    current_word_end = -1
                elif i > 0 and c != unprocessed_text[i-1]:
                    current_word += c
                    if current_word_start == -1:
                        current_word_start = i
                else:
                    pass

            texts.append(words)
        return texts
