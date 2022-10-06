from xpl.infrastructure.config import ConfigService

UNIT = 'xpl.infrastructure.storage'
config = ConfigService(UNIT)

from xpl.infrastructure.storage.repository import CloudStorageRepository
from xpl.infrastructure.storage.repository_async import Downloader
