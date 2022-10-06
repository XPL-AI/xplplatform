from xpl.infrastructure.config import ConfigService

UNIT = 'xpl.task'
config = ConfigService(UNIT)

from xpl.task.task_service import TaskService, Task, TaskNotFoundException, SetupTaskInvalidInputException
from xpl.task.modality_service import Modality, ModalityNotExist, ModalityService


