"""
Filename: /data/git/xplplatform/xpl/train/graphs/measurers/factory.py
Path: /data/git/xplplatform/xpl/train/graphs/measurers
Created Date: Tuesday, May 4th 2021, 2:32:20 pm
Author: Ali S. Razavian

Copyright (c) 2021 XPL Technologies AB
"""

from xpl.measurer.xpl_measurer import XPLMeasurer
from xpl.measurer.one_vs_rest import OneVsRest
from xpl.measurer.image.localization_measurer import LocalizationMeasurer


class measurerFactory:

    def get_measurer(self,
                     name: str,
                     definition: dict,
                     ) -> XPLMeasurer:
        measurer_type = definition['type']
        if measurer_type == 'one_vs_rest':
            return OneVsRest(name=name,
                             definition=definition)
        if measurer_type == 'image_recognition':
            return LocalizationMeasurer(name=name,
                                        definition=definition)

        else:
            raise BaseException(f'Unknown {measurer_type=} in {definition=}')
