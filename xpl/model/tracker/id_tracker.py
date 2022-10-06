import datetime
import uuid


class IDTracker:

    def __init__(self):

        
        self.__model_id = None
        self.__start_time = self.get_utc_time_now()
        self.__execution_id = self.generate_new_execution_id()

        self.__models: dict = {}

    def get_execution_id(self
                         ) -> str:
        return self.__execution_id

    def get_current_model_id(self
                             ) -> str:
        if self.__model_id is None:
            self.generate_new_model_id()
        return self.__model_id

    def generate_new_execution_id(self
                                  ) -> str:
        return f'execution_{self.__start_time}_{uuid.uuid4()}'

    def generate_new_model_id(self
                              ) -> str:
        self.__model_id = f'model_{self.get_utc_time_now()}_{uuid.uuid4()}'
        return

    def get_start_time(self
                       ) -> str:
        return self.__start_time

    def get_utc_time_now(self
                         ) -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    def monitor_models(self,
                       models: dict
                       ) -> None:
        self.__models = models