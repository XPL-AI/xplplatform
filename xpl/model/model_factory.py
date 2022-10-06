from xpl.model.neural_net.xpl_model import XPLModel
from xpl.model.neural_net.backbone.image_s import ImageS
from xpl.model.neural_net.backbone.audio_s import AudioS
from xpl.model.neural_net.heads.image.image_yolox import YoloX

class ModelFactory:

    def __init__(self,
                 ) -> None:
        pass

    def generate_model(self,
                       name: str,
                       definition: dict,
                       ) -> XPLModel:

        model_class = {
            ('backbone', 'image', 'onesize'): ImageS,
            ('backbone', 'audio', 'onesize'): AudioS,
            ('image_recognition', 'image', 'onesize'): YoloX,
        }

        model_type = definition['type']
        modality = definition['modality']
        model_size = definition['model_size']

        return model_class[(model_type, modality, model_size)](name=name,
                                                               definition=definition)