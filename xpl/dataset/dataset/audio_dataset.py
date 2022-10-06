####################################################################################################
# File: audio_dataset.py                                                                           #
# File Created: Thursday, 8th July 2021 10:32:19 am                                                #
# Author: Ali S. Razavian (ali@xpl.ai)                                                             #
#                                                                                                  #
# Last Modified: Wednesday, 15th September 2021 4:56:26 pm                                         #
# Modified By: Ali S. Razavian (ali@xpl.ai>)                                                       #
#                                                                                                  #
# Copyright 2020 - 2021 XPL Technologies AB, XPL Technologies AB                                   #
####################################################################################################

import torch
from xpl.dataset.augment.utils import load_audio_from_disk
from xpl.dataset.dataset.xpl_dataset import XPLDataset


class AudioDataset(XPLDataset):
    def __getitem__(self,
                    idx: int,
                    ) -> dict:
        data_point_id = self.all_data_points['data_point_id'].values[idx]

        data_point_local_file_name = self.all_data_points['data_point_local_file'].values[idx]

        assert isinstance(data_point_local_file_name, str), '\n'.join(f'{data_point_local_file_name=}',
                                                                      f'{type(data_point_local_file_name)=}',
                                                                      f'{idx=}',
                                                                      f'{self.all_data_points.index[idx]=}')
        target_name = list(self.targets.keys())[0]

        audio = load_audio_from_disk(data_point_local_file_name)

        audio_length = len(audio)

        center_x = int(self.all_data_points['center_x'].values[idx] * audio_length)
        #half_width = int(self.all_data_points['half_width'].values[idx] *  audio_length)
        half_width = 800

        start_x = center_x - half_width
        end_x = center_x + half_width
        pad_start = max(0, -start_x)
        pad_end = max(0, end_x - audio_length)
        
        start_x = max(0, start_x)
        end_x = min(end_x, audio_length)
        audio = torch.cat(
            [torch.rand(pad_start) * audio.std(),
             audio[start_x:end_x],
             torch.rand(pad_end) * audio.std()]
        ).unsqueeze(0)

        data_point = {'index': self.all_data_points.index[idx],
                      target_name: self.all_data_points[target_name].values[idx],
                      self.input_name: audio}
        return data_point

    def __calculate_informativeness(self,
                                    measurement: dict,
                                    sample: dict
                                    ) -> float:
        return 0.0
