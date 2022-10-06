from xpl.infrastructure.config import ConfigService

UNIT = 'xpl.annotation'
config = ConfigService(UNIT)

from xpl.annotation.entities import SimpleAnnotationJob, SimpleInstanceFrame, AnnotationJob
from xpl.annotation.annotation_service import AnnotationService
from xpl.annotation.annotation_service2 import AnnotationService2, AnnotationJob, AnnotationDataPoint
