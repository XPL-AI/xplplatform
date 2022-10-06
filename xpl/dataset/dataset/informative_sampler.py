"""
Filename: /data/git/xplplatform/xpl/data/data_exporter/dataset_sampler.py
Path: /data/git/xplplatform/xpl/data/data_exporter
Created Date: Monday, May 10th 2021, 10:52:33 am
Author: Ali S. Razavian

Copyright (c) 2021 XPL Technologies AB
"""
import numpy
import torch
import torch.utils.data

from typing import Iterator, List


class InformativeSampler(torch.utils.data.Sampler):

    def __init__(self,
                 log_informativeness_scores: numpy.array):
        self.__log_informativeness_scores = log_informativeness_scores
        self.__sampling_indices = self.__generate_sampling_indices(scores=numpy.copy(self.__log_informativeness_scores))

    def get_sampling_indices(self):
        return self.__sampling_indices

    def drop_sampling_indices(self, sample_indices_to_drop: List[int]):
        self.__sampling_indices = numpy.delete(self.__sampling_indices, sample_indices_to_drop)

    def __generate_sampling_indices(self,
                                    scores: numpy.array
                                    ) -> numpy.array:

        all_indices = numpy.array(range(len(scores)))
        informative_indices = all_indices[scores >= -1]
        other_indices = all_indices[scores < -1]
        other_indices_probability = numpy.exp(scores[other_indices])
        other_indices_probability /= (other_indices_probability.sum() + 1e-18)

        if len(informative_indices) > 0 and len(other_indices) > 0:
            num_easy_samples = int(min(len(informative_indices), len(other_indices)) * 1.4142)
            easy_indices = numpy.random.choice(a=other_indices,
                                               size=num_easy_samples,
                                               replace=True,
                                               p=other_indices_probability)
            indices = numpy.concatenate([informative_indices,  # we have mix of informative and easy samples
                                         easy_indices])
            sampling_indices = indices

        elif len(informative_indices) == 0:
            # Basically the training has ended!!
            sampling_indices = numpy.array([])

        else:
            sampling_indices = informative_indices

        numpy.random.shuffle(sampling_indices)
        return sampling_indices

    def __iter__(self
                 ) -> Iterator:
        return iter(self.__sampling_indices)

    def __len__(self
                ) -> int:
        return len(self.__sampling_indices)
