from xpl.task import config
from xpl.task.entities import Modality


class ModalityService:
    def list_modalities(self):
        modalities = []
        for modality in config['supported_modalities']:
            m = Modality(modality_id=modality)
            if modality in config['supported_model_sizes']:
                m.supported_model_sizes = config['supported_model_sizes'][modality]
            modalities.append(m)
        return modalities

    def get_modality(self, modality_id):
        if modality_id not in config['supported_modalities']:
            raise ModalityNotExist
        m = Modality(modality_id=modality_id)
        if modality_id in config['supported_model_sizes']:
            m.supported_model_sizes = config['supported_model_sizes']

        return m


class ModalityNotExist(Exception):
    pass
