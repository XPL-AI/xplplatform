import shutil

from os.path import join, exists

import numpy
import pandas
import torch
import torch.utils.data


from xpl.infrastructure.utils import timed


from xpl.dataset.dataset.image_dataset import ImageDataset
from xpl.dataset.dataset.audio_dataset import AudioDataset
from xpl.dataset.dataset.xpl_dataset import XPLDataset


class XPLDataLoader:
    """
    XPLDataLoader is the class that is responsible to load data.
    As an argument it receives a dictionary of datasets and number
    of threads that are necessary for loading
    """

    def __init__(self,
                 data_points: pandas.DataFrame,
                 background_data_points: pandas.DataFrame,
                 dataset_definition: dict,
                 modality: str,
                 batch_size: int = None,
                 num_workers: int = None,
                 ) -> None:

        # XPLDataLoader.__erase_directory(self.cache_dir)

        self.__dataset_definition = dataset_definition

        # TODO We should ask hardware service for these information
        self.__batch_size = batch_size if batch_size else 32
        self.__num_workers = num_workers if num_workers else 8

        self.__all_input_sets = set(data_points['input_set'])

        #assert 'train' in self.__all_input_sets, f'data_points does not have a training set: {self.__all_input_sets=}'
        #assert 'test' in self.__all_input_sets, f'data_points does not have a test set: {self.__all_input_sets=}'

        self.__datasets: dict[str, XPLDataset] = {}

        if 'targets' in self.__dataset_definition:
            self.targets = self.__dataset_definition['targets']

        self.targets = self.__dataset_definition['targets']
        for target_name, concept_ids in self.targets.items():
            data_points[target_name] = self.__concept_to_label(data_points['concept_id'],
                                                               concept_ids)

        self.__all_input_sets = self.__split_train_val_data_points(data_points)

        for input_set_name, data_subset in self.__all_input_sets.items():

            if modality == 'image':
                self.__datasets[input_set_name] = ImageDataset(input_set_name=input_set_name,
                                                               data_points=data_subset,
                                                               background_data_points=background_data_points,
                                                               modality=modality,
                                                               dataset_definition=dataset_definition,
                                                               )
            elif modality == 'audio':
                self.__datasets[input_set_name] = AudioDataset(input_set_name=input_set_name,
                                                               data_points=data_subset,
                                                               background_data_points=background_data_points,
                                                               modality=modality,
                                                               dataset_definition=dataset_definition,
                                                               )

        self.__loaders = {k: self.__reset_data_loaders(input_set=k)
                          for k in set(self.__all_input_sets.keys()) | {'training'}}

    def __split_train_val_data_points(self,
                                      data_points: pandas.DataFrame,
                                      split_ratio: float = 0.1,
                                      ) -> dict[str, pandas.DataFrame]:

        all_subset_names = list(set(data_points['input_set']))
        if 'train' in all_subset_names and 'val' in all_subset_names:
            return {
                'train': data_points[data_points['input_set'] == 'train'],
                'val': data_points[data_points['input_set'] == 'val'],
            }
        else:
            data_points['input_set'] = ''
            # 1- groupby lines to get samples
            # 2- count concept_ids and sort them from least to most frequent
            # 3- for the least common concept id, assign 10% of samples to the val set and the rest to the train set
            # 4- go the next least common concept_id and repeat 3
            grouped_data_points = data_points.groupby('data_point_id').groups

            while True:
                unassigned_data_points = data_points[data_points['input_set'] == '']
                unassigned_concept_count = unassigned_data_points['concept_id'].value_counts()
                assigned_to_train_concept_count = data_points[data_points['input_set'] == 'train']['concept_id'].value_counts()
                assigned_to_val_concept_count = data_points[data_points['input_set'] == 'val']['concept_id'].value_counts()
                if unassigned_concept_count.empty:
                    break

                selected_concept_id = unassigned_concept_count.index[-1]

                assigned_to_train = 0 if selected_concept_id not in assigned_to_train_concept_count.index else assigned_to_train_concept_count[
                    selected_concept_id]
                assigned_to_val = 0 if selected_concept_id not in assigned_to_val_concept_count.index else assigned_to_val_concept_count[
                    selected_concept_id]
                unassigned_count = unassigned_concept_count[selected_concept_id]
                need_to_move_to_eval = max(0, int((assigned_to_train + assigned_to_val + unassigned_count) * split_ratio - assigned_to_val))

                data_points_with_this_concept = unassigned_data_points[unassigned_data_points['concept_id'] == selected_concept_id]

                should_be_assigned_to_train_indices = []
                should_be_assigned_to_val_indices = []
                current_grouped_data_points = data_points_with_this_concept.groupby('data_point_id').groups
                current_sample_counter = 0
                for k in current_grouped_data_points.keys():
                    v = grouped_data_points[k]
                    if current_sample_counter < need_to_move_to_eval:
                        should_be_assigned_to_val_indices += v.values.tolist()
                    else:
                        should_be_assigned_to_train_indices += v.values.tolist()

                    current_sample_counter += len(current_grouped_data_points[k])

                data_points.loc[numpy.array(should_be_assigned_to_train_indices), 'input_set'] = 'train'
                data_points.loc[numpy.array(should_be_assigned_to_val_indices), 'input_set'] = 'val'

        return {
            'train': data_points[data_points['input_set'] == 'train'],
            'val': data_points[data_points['input_set'] == 'val'],
        }

    def __concept_to_label(self,
                           concept_series: pandas.Series,
                           concept_ids: list[str]
                           ) -> pandas.Series:
        concept_dict = {}
        for i, concept in enumerate(concept_ids):
            concept_dict[concept] = i
        label_series = concept_series.map(concept_dict, na_action='ignore')
        return label_series

    def get_input_sets(self):
        print(f'=========={list(self.__all_input_sets)}==========')
        return list(self.__all_input_sets)

    @timed
    def get_next_batch(self,
                       input_set: str
                       ) -> dict:
        try:
            batch = next(self.__loaders[input_set])
        except StopIteration:
            self.__loaders[input_set] = self.__reset_data_loaders(input_set=input_set)
            batch = None

        return batch

    @timed
    def record_predictions_and_measurements(self,
                                            input_set: str,
                                            measurements: dict,
                                            batch: dict,
                                            ) -> None:
        #  We skip recording predictions during training since we validate on the train set later on.
        if input_set == 'training':
            return None

        self.__datasets[input_set].record_predictions_and_measurements(measurements=measurements,
                                                                       batch=batch)
        return None

    def __reset_data_loaders(self,
                             input_set: str,
                             ):
        if input_set == 'training':
            dataset = self.__datasets['train']
            sampler = dataset.generate_informative_sampler()
        else:
            dataset = self.__datasets[input_set]
            sampler = dataset.generate_iterative_sampler()

        return iter(torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=self.__batch_size,
                                                sampler=sampler,
                                                num_workers=1,
                                                drop_last=input_set=='training',
                                                collate_fn=self.__collate_function
                                                ))

    def __collate_function(self,
                           data_list: list[dict]
                           ) -> dict:
        """
        This function takes a list of nested dictionaries with numbers, numpy arrays or etc as values.
        and converts them to the nested dictionary format (as it was the case )

        Args:
            data_list (list[dict]): [description]

        Raises:
            BaseException: [description]
            BaseException: [description]

        Returns:
            dict: [description]
        """

        data = {k: [data_point[k]
                    for data_point in data_list
                    if data_point[k] is not None]
                for k in data_list[0]}
        ret_data = {}
        for k, v in data.items():
            if isinstance(v[0], str):
                ret_data[k] = v

            elif isinstance(v[0], (int, numpy.int64)):
                ret_data[k] = torch.LongTensor(numpy.array(v, dtype=int))

            elif isinstance(v[0], float):
                ret_data[k] = torch.FloatTensor(numpy.array(v))

            elif isinstance(v[0], numpy.ndarray):
                ret_data[k] = numpy.zeros(
                    (len(v), *numpy.array([x.shape for x in v]).max(axis=0).tolist()), dtype=v[0].dtype)
                if (v[0].ndim == 1):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-1]] = v[i]
                elif (v[0].ndim == 2):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-2], :v[i].shape[-1]] = v[i]
                elif (v[0].ndim == 3):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-3], v[i].shape[-2], :v[i].shape[-1]] = v[i]
                else:
                    raise BaseException(f'too many dimension for k {k} and v {v.ndim=}')
                
            elif isinstance(v[0], torch.Tensor):
                if isinstance(v[0], torch.FloatTensor):
                    tensor_type = torch.FloatTensor
                elif isinstance(v[0], torch.LongTensor):
                    tensor_type = torch.LongTensor
                else:
                    raise BaseException(f'Unknown tensor type: {type(v[0])}')

                ret_data[k] = tensor_type(len(v),
                                          *numpy.array([list(x.shape) for x in v]).max(axis=0).tolist()
                                          ).fill_(0)
                if (v[0].ndim == 1):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-1]] = v[i]
                elif (v[0].ndim == 2):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-2], :v[i].shape[-1]] = v[i]
                elif (v[0].ndim == 3):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-3], :v[i].shape[-2], :v[i].shape[-1]] = v[i]
                elif (v[0].ndim == 4):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-4], :v[i].shape[-3], :v[i].shape[-2], :v[i].shape[-1]] = v[i]
                elif (v[0].ndim == 5):
                    for i in range(len(v)):
                        ret_data[k][i, :v[i].shape[-5], :v[i].shape[-4], :v[i].shape[-3], :v[i].shape[-2], :v[i].shape[-1]] = v[i]
                else:
                    raise BaseException(f'too many dimension for k {k} and v {v.ndim=}')

            else:
                raise BaseException(f'unknown type {type(v[0])} for k {k} and v {v}')

        return ret_data
