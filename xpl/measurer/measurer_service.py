from xpl.measurer.measurer_factory import measurerFactory
from xpl.measurer.xpl_measurer import XPLMeasurer

class measurerService:

    def __init__(self
                 ) -> None:
        self.__measurer_factory = measurerFactory()

    def get_measurers(self,
                      definitions: dict,
                      ) -> dict:
        measurers = {}
        for measurer_name, definition in definitions.items():
            measurers[measurer_name] = self.get_measurer(name=measurer_name,
                                                         definition=definition)
        return measurers

    def get_measurer(self,
                     name: str,
                     definition: dict,
                     ) -> XPLMeasurer:
        return self.__measurer_factory.get_measurer(name=name,
                                                    definition=definition)
