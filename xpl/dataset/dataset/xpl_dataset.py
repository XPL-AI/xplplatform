import logging
import os
import abc
import time
from augment.data_augmenter import DataAugmenter
import numpy
from numpy.core.numeric import indices
import torch
import torch.utils.data
import pandas
from collections import defaultdict

from typing import Optional, Tuple, List

from xpl.data import config
from xpl.infrastructure.storage.repository_async import Downloader
from xpl.dataset.augment.image_augmenter import ImageAugmenter
from xpl.dataset.dataset.informative_sampler import InformativeSampler

logger = logging.getLogger(__name__)


class XPLDataset(torch.utils.data.Dataset):
    def __init__(self,
                 input_set_name: str,
                 data_points: pandas.DataFrame,
                 background_data_points: pandas.DataFrame,
                 modality: str,
                 dataset_definition: dict,
                 ) -> None:

        super(XPLDataset, self).__init__()
        self.cache_dir = os.path.join(config.DATA_DIR, 'cache')

        self.input_set_name = input_set_name

        self.data_points = data_points.copy()  # The is a reference to the datapoints. We shouldn't mess up the original
        self.data_points.reset_index(drop=True, inplace=True)
        self.data_points = self.__download_data_points(self.data_points)

        if background_data_points is not None:
            self.background_data_points = background_data_points.copy()  # The is a reference to the background datapoints.
            self.background_data_points.reset_index(drop=True, inplace=True)
            self.background_data_points = self.__download_data_points(self.background_data_points)

        self.__modality = modality

        self.__epoch_summary = []
        self.__epoch_summary_df = pandas.DataFrame({})
        self.__dataset_definition = dataset_definition

        self.input_name = self.__dataset_definition['input_name']
        self.__input_size = self.__dataset_definition['input_size']
        self.__output_size = None
        self.has_targets = self.__dataset_definition['has_targets']  # TODO do we need this?

        if 'targets' in self.__dataset_definition:
            self.targets = self.__dataset_definition['targets']
            self.__output_size = self.__dataset_definition['output_size']

        self.augment = self.init_data_augmenter(input_size=self.__input_size,
                                                output_size=self.__output_size,
                                                background_data_points=self.background_data_points)

        self.grouped_data_points = self.data_points.groupby('data_point_id')
        self.grouped_data_points_dict = self.grouped_data_points.groups
        self.grouped_data_points_list = [[k, v] for k, v in self.grouped_data_points_dict.items()]

    def __len__(self
                ) -> int:
        return len(self.grouped_data_points_list)

    @abc.abstractmethod
    def calculate_informativeness(self,
                                  measurement: dict,
                                  sample: dict,
                                  ) -> Tuple[float, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def init_data_augmenter(self,
                            input_size: dict[str, int],
                            output_size: dict[str, int],
                            background_data_points: Optional[pandas.DataFrame],
                            ) -> DataAugmenter:
        raise NotImplementedError

    def record_predictions_and_measurements(self,
                                            measurements: dict,
                                            batch: dict,
                                            ) -> None:
        indices = batch['index'].cpu().numpy()
        measurement_summary = self.__summarize_measurement(indices=indices,
                                                           measurements=measurements)
        # #predictions = {target: batch[f'pred_{target}'] for target in self.targets.keys()}
        # data_distribution_summary = self.__summarize_batch_distribution(indices=indices,
        #                                                                 predictions=predictions)

        self.__epoch_summary.append(measurement_summary)

    def generate_informative_sampler(self
                                     ) -> InformativeSampler:

        informativeness_scores = self.__compile_informative_scores()

        return InformativeSampler(informativeness_scores)

    def generate_iterative_sampler(self,
                                   ) -> torch.utils.data.SequentialSampler:
        return torch.utils.data.SequentialSampler(self.grouped_data_points_list)
    
    def report_sample_is_corrupt(self, index):
        # TODO make sure this datapoint never shows up again!
        data_point_id, indices_in_dataframe = self.grouped_data_points_list[index]
        data_point_local_file_name = self.data_points['data_point_local_file'].values[indices_in_dataframe[0]]
        logger.warn(f'DELETE {data_point_local_file_name}')
        os.remove(data_point_local_file_name)

    def __summarize_batch_distribution(self,
                                       indices: numpy.array,
                                       predictions: numpy.ndarray,
                                       ) -> dict:
        # TODO update the prediction distribution
        # for k, v in predictions.items():
        #     print(k, v.shape)
        return {}

    def __compile_informative_scores(self,
                                     ) -> numpy.array:
        # TODO compile informative samples
        import matplotlib.pylab as plt
        if len(self.__epoch_summary) > 0:
            self.__epoch_summary_df = pandas.concat(self.__epoch_summary).sort_index()
            self.__epoch_summary_df.sum(axis=1)

            self.data_points['log_informativeness'] += numpy.log(self.__epoch_summary_df.sum(axis=1))
            self.grouped_data_points = self.data_points.groupby('data_point_id')

            #plt.hist(numpy.log(self.__epoch_summary_df.sum(axis=1).values), bins=100)
            # plt.title(self.__input_set_name)
            # plt.show()
            self.__epoch_summary = []

        return self.grouped_data_points['log_informativeness'].sum().values

    def __summarize_measurement(self,
                                indices: numpy.array,
                                measurements: dict,
                                ) -> list[dict]:

        all_samples = {'index': indices}
        for measurement_type, measurement in measurements.items():
            for measurement_name, v in measurement.items():
                
                all_samples |= {(measurement_type, measurement_name): self.__detached_variable(v)}

        return pandas.DataFrame(all_samples).set_index(['index'])


    def __detached_variable(self, v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        else:
            return v

    def __download_data_points(self,
                               data_points: pandas.DataFrame
                               ) -> None:
        gsutil_uri_list_series: pandas.Series = data_points['data_point_file']

        file_to_indices = {}
        for index, value in gsutil_uri_list_series.items():
            if value in file_to_indices:
                file_to_indices[value].append(index)
            else:
                file_to_indices[value] = [index]

        gsutil_uri_list = list(gsutil_uri_list_series.unique())

        ok, failed = Downloader.download(uri_list=gsutil_uri_list,
                                         destination_dir=self.cache_dir,
                                         number_of_processes=8,
                                         coroutines_batch_size=512,
                                         print_progress=True,
                                         accept_fail_rate=0.2,
                                         do_not_download_if_file_exists=True)

        all_ok_indices = []
        all_ok_files = []
        for key in ok.keys():
            indices = file_to_indices[key]
            for i in indices:
                all_ok_indices.append(i)
                all_ok_files.append(ok[key].file_name)

        all_failed_indices = []
        for key in failed.keys():
            indices = file_to_indices[key]
            for i in indices:
                all_failed_indices.append(i)

        for ok_file in all_ok_files:
            assert isinstance(ok_file, str), f'{ok_file=} must be string but is {type(ok_file)=}'

        data_point_files = pandas.Series(data=all_ok_files, index=all_ok_indices, name='data_point_local_file')
        for f in data_point_files:
            assert isinstance(f, str), f'{f=}'

        data_points = pandas.concat([data_points, data_point_files], axis=1)

        for f in data_points['data_point_local_file']:
            assert isinstance(f, str), f'{f=}'

        if len(all_failed_indices) > 0:
            self.__drop_samples(all_failed_indices)

        return data_points

    def __drop_samples(self, sample_indices_to_drop: List[int]):
        self.data_points.drop(sample_indices_to_drop)
        self.__sampler.drop_sampling_indices(sample_indices_to_drop)

    @classmethod
    def __erase_directory(cls, directory):
        if exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
