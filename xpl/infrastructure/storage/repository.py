from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import exists, isfile, join, getsize
from typing import List

import io
import multiprocessing
import ntpath
import os

import requests

import pandas

from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.api_core import page_iterator
from google.cloud.storage import Blob

from xpl.infrastructure.storage import config

CLIENT: storage.Client = None


class CloudStorageRepository:
    def __init__(self,
                 bucket_name=None,
                 credentials_file_path=None):
        self.__bucket_name = bucket_name
        self.__credentials_file_path = credentials_file_path

        global CLIENT
        if CLIENT is None:
            if credentials_file_path is not None:
                if exists(credentials_file_path):
                    CLIENT = storage.Client.from_service_account_json(credentials_file_path)
                elif exists(join(config.CONFIGS_DIR, credentials_file_path)):
                    CLIENT = storage.Client.from_service_account_json(join(config.CONFIGS_DIR, credentials_file_path))
                else:
                    CLIENT = storage.Client()
            elif 'google_cloud_storage_service_account_file' in config:
                if exists(config['google_cloud_storage_service_account_file']):
                    CLIENT = storage.Client.from_service_account_json(config['google_cloud_storage_service_account_file'])
                elif exists(join(config.CONFIGS_DIR, config['google_cloud_storage_service_account_file'])):
                    CLIENT = storage.Client.from_service_account_json(
                        join(config.CONFIGS_DIR, config['google_cloud_storage_service_account_file']))
                else:
                    CLIENT = storage.Client()
            else:
                CLIENT = storage.Client()

    def get_bucket_details(self,
                           bucket_name=None) -> storage.bucket.Bucket:
        if bucket_name is None:
            return CLIENT.get_bucket(self.__bucket_name)
        return CLIENT.get_bucket(bucket_name)

    def bucket_exist(self, bucket_name) -> bool:
        try:
            CLIENT.get_bucket(bucket_name)
        except NotFound:
            return False

        return True

    def create_bucket(self,
                      bucket_name,
                      location: str = "EUROPE-NORTH1"):
        bucket = CLIENT.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        new_bucket = CLIENT.create_bucket(bucket, location=location)

        return new_bucket

    def upload_from_file(self,
                         file_path,
                         blob_name,
                         metadata=None,
                         bucket_name=None
                         ) -> str:
        content_type = 'application/jpg'

        metadata = metadata or {}

        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        bucket = self.__get_bucket(bucket_name)

        blob = bucket.blob(blob_name)

        blob.content_type = content_type
        blob.metadata = metadata

        with open(file_path, "rb") as f:
            blob.upload_from_file(f)

        return f'gs://{bucket_name}/{blob_name}'

    def upload_from_stream(self,
                           stream,
                           blob_name,
                           metadata=None,
                           bucket_name=None):
        metadata = metadata or {}

        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        bucket = self.__get_bucket(bucket_name)

        blob = bucket.blob(blob_name)
        blob.metadata = metadata

        blob.upload_from_file(stream)
        return f'gs://{bucket_name}/{blob_name}'

    def upload_from_dataframe(self,
                              dataframe: pandas.DataFrame,
                              blob_name: str,
                              metadata=None,
                              bucket_name=None):
        metadata = metadata or {}

        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        bucket = self.__get_bucket(bucket_name)

        blob = bucket.blob(blob_name)
        blob.metadata = metadata

        byte_stream = io.BytesIO()
        dataframe.to_csv(byte_stream)
        byte_stream.seek(0)

        blob.upload_from_file(byte_stream)

    def get_url_for_upload(self,
                           blob_name: str,
                           bucket_name=None):
        """Returns URL valid for 60 seconds, that http client uses to upload data."""
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        bucket = self.__get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return blob.create_resumable_upload_session(), f'gs://{bucket_name}/{blob_name}'

    def list_blobs(self,
                   prefix,
                   ends_with='',
                   bucket_name=None):
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name

        blobs = CLIENT.list_blobs(self.__get_bucket(bucket_name), 100, prefix=f'{prefix}')

        bls = list(blobs)
        names = []
        for b in bls:
            if b.name.endswith(ends_with):
                names.append(b.name)

        return names

    def list_directories(self,
                         prefix,
                         starts_with='',
                         bucket_name=None,
                         sort_reverse=False,
                         show_full_path=False):
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        if not prefix.endswith('/'):
            prefix += '/'
        extra_params = {
            "projection": "noAcl",
            "prefix": prefix,
            "delimiter": '/'
        }

        path = "/b/" + bucket_name + "/o"
        iterator = page_iterator.HTTPIterator(
            client=CLIENT,
            api_request=CLIENT._connection.api_request,
            path=path,
            items_key='prefixes',
            item_to_value=(lambda it, value: value),
            extra_params=extra_params,
        )

        results = []
        for directory in iterator:
            if directory.split('/')[-2].startswith(starts_with):
                if show_full_path:
                    results.append(directory)
                else:
                    results.append(directory.split('/')[-2])

        return sorted(results, reverse=sort_reverse)

    def get_url_for_download(self,
                             blob_name,
                             time_to_live_seconds=60,
                             bucket_name=None):
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        bucket = self.__get_bucket(bucket_name)

        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(datetime.utcnow() + timedelta(seconds=time_to_live_seconds))

        return url

    def get_url_for_download_by_uri(self,
                                    uri: str,
                                    time_to_live_seconds=60):
        blob = storage.Blob.from_string(uri, client=CLIENT)
        url = blob.generate_signed_url(datetime.utcnow() + timedelta(seconds=time_to_live_seconds))

        return url

    def download_to_file(self,
                         blob_name,
                         destination_file_name,
                         bucket_name=None):
        """Downloads a blob from storage and put in file on local disk."""
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        blob: Blob = self.__get_bucket(bucket_name=bucket_name).blob(blob_name)

        directory = os.path.dirname(destination_file_name)
        os.makedirs(directory, exist_ok=True)

        blob.download_to_filename(destination_file_name)

    def download_to_file_by_uri(self,
                                uri,
                                destination_file_name):
        """Downloads a blob from storage using uri and put in file on local disk."""
        blob = storage.Blob.from_string(uri, client=CLIENT)

        directory = os.path.dirname(destination_file_name)
        os.makedirs(directory, exist_ok=True)

        blob.download_to_filename(destination_file_name)

    def download_as_text(self,
                         blob_name,
                         bucket_name=None):
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        blob = self.__get_bucket(bucket_name=bucket_name).blob(blob_name)

        return blob.download_as_text()

    def download_as_dataframe(self,
                              blob_name: str,
                              bucket_name: str = None) -> pandas.DataFrame:
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name

        blob = self.__get_bucket(bucket_name=bucket_name).blob(blob_name)
        bytes_data = blob.download_as_bytes()

        return pandas.read_csv(io.BytesIO(bytes_data))

    def download_as_bytes(self,
                          blob_name,
                          bucket_name=None) -> bytes:
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name

        blob = self.__get_bucket(bucket_name=bucket_name).blob(blob_name)
        bytes_data = blob.download_as_bytes()

        return bytes_data

    def download_as_bytes_by_uri(self,
                                 uri: str) -> bytes:
        blob = storage.Blob.from_string(uri, client=CLIENT)
        bytes_data = blob.download_as_bytes()

        return bytes_data

    def get_update_time(self,
                        blob_name,
                        bucket_name=None):
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name
        blob = self.__get_bucket(bucket_name=bucket_name).get_blob(blob_name=blob_name)

        return blob.updated

    def exists(self,
               blob_name,
               bucket_name=None):
        bucket_name = self.__bucket_name if bucket_name is None else bucket_name

        return self.__get_bucket(bucket_name=bucket_name).blob(blob_name).exists()

    def __get_bucket(self,
                     bucket_name) -> storage.bucket.Bucket:
        return CLIENT.bucket(bucket_name)


class DownloadItem:
    def __init__(self, local_file_path: str, cloud_file_uri: str):
        self.cloud_file_uri = cloud_file_uri
        self.local_file_path = local_file_path


class DownloadItemResult:
    def __init__(self,
                 result: str,
                 size: int,
                 error=None):
        self.result = result
        self.size = size
        self.error = error


class Downloader:
    def __init__(self,
                 cloud_file_uris: List[str],
                 local_directory: str,
                 number_of_processes: int = None):
        #
        # # Check if bucket exists and we have access
        # if not self.__repository.bucket_exist(bucket):
        #     raise Exception(f'{bucket=} does not exist.')

        self.local_directory = local_directory
        self.cloud_file_uris = cloud_file_uris
        self.number_of_processes = multiprocessing.cpu_count() * 2 if number_of_processes is None else number_of_processes

    def __enter__(self):
        self.pool = Pool(self.number_of_processes)
        self.download_items: List[DownloadItem] = []

        for f in self.cloud_file_uris:
            file_name = f.replace('gs://', '')
            self.download_items.append(DownloadItem(local_file_path=os.path.join(self.local_directory, file_name),
                                                    cloud_file_uri=f))
        self.imap_iterator = self.pool.imap(download, self.download_items, chunksize=25)

        return self

    def __iter__(self):
        return self.imap_iterator.__iter__()

    def __exit__(self, one, two, three):
        self.pool.close()


def download(download_item: DownloadItem) -> DownloadItemResult:
    try:
        repository = CloudStorageRepository()
        repository.download_to_file_by_uri(uri=download_item.cloud_file_uri,
                                           destination_file_name=download_item.local_file_path)

    #     There should be a strategy around failed downloads:
    #       This is not a one time job. As exact error types and root reasons are understood - appropriate action can be taken:
    #       - Retries with exponential back-off
    #       - put to the end of the queue to retry again, when all other items were attempted already
    #       - give up and skip the file
    #       - stop the whole thing, log and alert CRITICAL failure
    except requests.exceptions.ReadTimeout as e:
        return DownloadItemResult(result=e.args[0], size=0, error=str(e))

    except Exception as unknown_error:
        return DownloadItemResult(result=unknown_error.args[0], size=0, error=str(unknown_error))

    return DownloadItemResult(result='OK', size=os.path.getsize(download_item.local_file_path))


class UploadItem:
    def __init__(self,
                 local_file_path: str,
                 cloud_file_path: str,
                 repository: CloudStorageRepository):
        self.cloud_file_path = cloud_file_path
        self.local_file_path = local_file_path
        self.repository = repository


class UploadItemResult:
    def __init__(self, result: str, size: int):
        self.result = result
        self.size = size


class Uploader:
    def __init__(self,
                 bucket: str,
                 directory: str,
                 files_to_upload: List[str],
                 number_of_processes: int = None):
        self.__repository = CloudStorageRepository(bucket_name=bucket)

        # Check if bucket exists an we have access
        if not self.__repository.bucket_exist(bucket):
            raise Exception(f'{bucket=} does not exist.')

        # Verify that files requested exist on the disk
        #   At the same time sum the volume of data
        non_existing_files = []
        self.total_upload_volume = 0
        for file in files_to_upload:
            if not isfile(file):
                non_existing_files.append(file)
            file_volume = getsize(file)
            self.total_upload_volume += file_volume

        if len(non_existing_files) > 0:
            raise Exception(f'{len(non_existing_files)} files out of {len(files_to_upload)} were not found on the disk.')

        self.bucket = bucket
        self.directory = directory
        self.files_to_upload = files_to_upload
        self.number_of_processes = multiprocessing.cpu_count() * 2 if number_of_processes is None else number_of_processes

    def __enter__(self):
        self.pool = Pool(self.number_of_processes)
        self.upload_items: List[UploadItem] = []
        for f in self.files_to_upload:
            file_name = path_leaf(f)
            self.upload_items.append(UploadItem(local_file_path=f,
                                                cloud_file_path=join(self.directory, file_name[0], file_name[1], file_name[2], file_name),
                                                repository=self.__repository))
        self.imap_iterator = self.pool.imap(upload, self.upload_items, chunksize=25)

        return self

    def __iter__(self):
        return self.imap_iterator.__iter__()

    def __exit__(self, one, two, three):
        print(f'{one}\n{two}\n{three}')
        self.pool.close()


def upload(upload_item: UploadItem) -> UploadItemResult:
    size = getsize(upload_item.local_file_path)
    try:
        upload_item.repository.upload_from_file(upload_item.local_file_path, upload_item.cloud_file_path)
    except Exception as e:
        return UploadItemResult(result=e.args[0], size=size)

    return UploadItemResult(result='OK', size=size)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
