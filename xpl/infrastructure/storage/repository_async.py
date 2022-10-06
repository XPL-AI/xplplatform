import asyncio
import os
import sys
import time

from multiprocessing import Pool
from os.path import join, dirname, exists, getsize
from os import makedirs, getpid
from typing import List, Dict

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request

import aiohttp
import aiohttp.client_exceptions
from tqdm import tqdm, trange

from xpl.infrastructure.storage import config


CLOUD_API_URL = 'https://storage.googleapis.com/'

SCOPE = (
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/devstorage.read_only",
    "https://www.googleapis.com/auth/devstorage.read_write",
)

"""The scopes required for authenticating as a Cloud Storage consumer."""
if 'google_cloud_storage_service_account_file' in config:
    SERVICE_ACCOUNT_FILE_PATH = join(config.CONFIGS_DIR, config['google_cloud_storage_service_account_file'])
else:
    SERVICE_ACCOUNT_FILE_PATH = join(config.CONFIGS_DIR, '.json')

CREDENTIALS: Credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE_PATH, scopes=SCOPE)


class DownloadBatch:
    def __init__(self,
                 uri_batch: List[str],
                 destination_dir):
        self.destination_dir = destination_dir
        self.gsutil_uri_batch = uri_batch


class DownloadResult:
    def __init__(self, uri: str, status: str, size: int, downloaded_bytes: bytes = None, file_name: str = None):
        self.uri = uri
        self.status = status
        self.size = size
        self.downloaded_bytes = downloaded_bytes
        self.file_name = file_name


class DownloadBatchResult:
    def __init__(self, results: Dict[str, DownloadResult]):
        self.results = results


class Downloader:
    def __init__(self,
                 uri_list: List[str],
                 destination_dir: str = None,
                 number_of_processes: int = 1,
                 coroutines_batch_size: int = 100):
        validate_uri_list(uri_list)

        self.destination_dir = destination_dir
        self.number_of_processes = number_of_processes

        self.uri_batches = make_batches(uri_list, n=coroutines_batch_size)
        self.download_batches: List[DownloadBatch] = []
        for b in self.uri_batches:
            self.download_batches.append(DownloadBatch(b, self.destination_dir))

    def __enter__(self):
        self.pool = Pool(self.number_of_processes)
        if self.destination_dir is None:
            self.imap_iterator = self.pool.imap(download_batch_to_memory_wrapper, self.download_batches, chunksize=1)
        else:
            self.imap_iterator = self.pool.imap(download_batch_to_dir_wrapper, self.download_batches, chunksize=1)
        return self

    def __iter__(self):
        return self.imap_iterator.__iter__()

    def __exit__(self, one, two, three):
        # print(f'{one}\n{two}\n{three} pool closed')
        self.pool.close()

    @classmethod
    def download(cls,
                 uri_list: List[str],
                 destination_dir: str = None,
                 number_of_processes: int = 1,
                 coroutines_batch_size: int = 100,
                 print_progress=True,
                 accept_fail_rate=0.01,
                 do_not_download_if_file_exists=False,
                 download_label: str = None):
        with tqdm(total=len(uri_list)) as progress_bar:

            already_exist = {}
            if do_not_download_if_file_exists is True and destination_dir is not None:
                already_exist, uri_list = filter_out_existing_files(uri_list, destination_dir)

            ok = {}
            ok_bytes = 0
            failed = {}
            total_processed = 0
            start = time.perf_counter()

            total_to_download = len(uri_list)
            # print(f'Downloading {total_to_download} files:')
            time.sleep(0.1)
            Downloader.__update_progress(progress_bar=progress_bar,
                                         n=len(already_exist),
                                         status='Downloading',
                                         files_ok=len(ok),
                                         bytes_ok=ok_bytes,
                                         files_failed=len(failed),
                                         elapsed_seconds=time.perf_counter() - start)
            if total_to_download > 0:
                with Downloader(uri_list=uri_list,
                                destination_dir=destination_dir,
                                number_of_processes=number_of_processes,
                                coroutines_batch_size=coroutines_batch_size) \
                        as downloader:

                    batch_count = 0
                    for download_batch_result in downloader:
                        for result in download_batch_result.results.values():
                            if result.status == '200':
                                ok[result.uri] = result
                                ok_bytes += result.size
                            else:
                                failed[result.uri] = result
                        total_processed += len(download_batch_result.results)
                        fail_rate = len(failed) / total_processed
                        if fail_rate > accept_fail_rate:
                            raise Exception(
                                f'Too many files failed to download from the storage. failed_count={len(failed)} fail_rate={fail_rate * 100}%')

                        # if print_progress and batch_count % 5 == 0:
                        #     elapsed_time = time.perf_counter() - start
                        #     total_remaining = total_to_download - total_processed
                        #     estimated_time_left = total_remaining * elapsed_time / total_processed
                        #
                        #     Downloader.__print_progress('Downloading..', len(ok), ok_bytes, len(failed), elapsed_time, estimated_time_left,
                        #                                 download_label=download_label)
                        # batch_count += 1

                        Downloader.__update_progress(progress_bar=progress_bar,
                                                     n=len(download_batch_result.results),
                                                     status='Downloading',
                                                     files_ok=len(ok),
                                                     bytes_ok=ok_bytes,
                                                     files_failed=len(failed),
                                                     elapsed_seconds=time.perf_counter() - start)

            # Downloader.__print_progress('Download finished', len(ok), ok_bytes, len(failed), time.perf_counter() - start, 0, print_in_place=False,
            #                             already_existed=len(already_exist), download_label=download_label)
            Downloader.__update_progress(progress_bar=progress_bar,
                                         n=0,
                                         status='Downloading',
                                         files_ok=len(ok),
                                         bytes_ok=ok_bytes,
                                         files_failed=len(failed),
                                         elapsed_seconds=time.perf_counter() - start)
            ok = dict(ok, **already_exist)

            return ok, failed

    @classmethod
    def __print_progress(cls, status, ok, ok_bytes, failed, elapsed, remaining, print_in_place=True, already_existed=0,
                         download_label: str = ''):
        speed = ok_bytes / elapsed
        speed_in_files = ok / elapsed
        if print_in_place:
            sys.stdout.write("%s: [%s]  "
                             "OK: [%d]  MB OK: [%d]  FAILED: [%d]  Elapsed: [%d]s, Remaining: [%d]s  AVG Speed: [%f]MB/s [%d]files/s \r" %
                             (download_label, status, ok, ok_bytes / 1024 / 1024, failed, elapsed, remaining, speed / 1024 / 1024, speed_in_files))
            sys.stdout.flush()
        else:
            print("%s: [%s]  OK: [%d]  MB OK: [%d]  "
                  "FAILED: [%d]  EXISTED: [%d]  Elapsed: [%d]s, Remaining: [%d]s  AVG Speed: [%f]MB/s [%d]files/s \r" %
                  (download_label, status, ok, ok_bytes / 1024 / 1024, failed, already_existed, elapsed, remaining, speed / 1024 / 1024, speed_in_files))

    @classmethod
    def __update_progress(cls, progress_bar: tqdm,
                          n: int,
                          status,
                          files_ok,
                          bytes_ok,
                          files_failed,
                          elapsed_seconds):
        speed_in_bytes = bytes_ok / elapsed_seconds
        speed_in_files = files_ok / elapsed_seconds

        progress_bar.set_description(status)
        progress_bar.set_postfix_str(s=f'{files_ok}Files  '
                                       f'{bytes_ok / 1024 /1024 :.2f}MB  '
                                       f'{speed_in_files :.2f}Files/s '
                                       f'{speed_in_bytes / 1024 / 1024 :.2f}MB/s '
                                       f'{files_failed}FilesFailed')

        progress_bar.update(n)
        progress_bar.refresh()


class SingleProcessDownloader:
    def __init__(self,
                 uri_list: List[str],
                 destination_dir: str = None,
                 coroutines_batch_size: int = 100):
        validate_uri_list(uri_list)

        self.destination_dir = destination_dir

        self.gsutil_uri_batches = make_batches(uri_list, n=coroutines_batch_size)
        self.download_batches: List[DownloadBatch] = []
        for b in self.gsutil_uri_batches:
            self.download_batches.append(DownloadBatch(b, self.destination_dir))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if len(self.download_batches) > self.n:
            next_batch = self.download_batches[self.n]
            self.n += 1
            if self.destination_dir is None:
                return download_batch_to_memory_wrapper(next_batch)
            else:
                return download_batch_to_dir_wrapper(next_batch)
        raise StopIteration()


async def download_to_dir(gsutil_uri,
                          destination_dir,
                          session: aiohttp.ClientSession = None):
    download_result: DownloadResult
    if session is None:
        async with aiohttp.ClientSession() as session:
            download_result: DownloadResult = await download_to_memory(gsutil_uri, session)
    else:
        download_result: DownloadResult = await download_to_memory(gsutil_uri, session)

    if download_result.status == '200':
        bucket, resource_name = parse_blob_uri(gsutil_uri)
        destination_file_name = get_local_file_path(destination_dir, bucket, resource_name)
        directory = dirname(destination_file_name)
        makedirs(directory, exist_ok=True)
        with open(destination_file_name, 'wb') as file:
            file.write(download_result.downloaded_bytes)
            download_result.downloaded_bytes = None
            download_result.file_name = destination_file_name

    return download_result


async def download_to_memory(gsutil_uri,
                             session: aiohttp.ClientSession = None) -> DownloadResult:
    url = __make_cloud_api_get_url(gsutil_uri)
    if session is None:
        async with aiohttp.ClientSession() as session:
            try:
                headers = __get_authorization_headers()
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        downloaded_bytes: bytes = await response.read()
                        downloaded_bytes_len = len(downloaded_bytes)
                        return DownloadResult(uri=gsutil_uri,
                                              status=str(response.status),
                                              size=downloaded_bytes_len,
                                              downloaded_bytes=downloaded_bytes)
                    else:
                        return DownloadResult(uri=gsutil_uri, status=str(response.status), size=0)
            except aiohttp.client_exceptions.ClientConnectorError:
                # https://stackoverflow.com/questions/57046073/why-do-i-get-the-error-clientconnectorerror
                return DownloadResult(uri=gsutil_uri, status=str(900), size=0)
            except asyncio.exceptions.TimeoutError:
                return DownloadResult(uri=gsutil_uri, status=str(901), size=0)
    else:
        try:
            headers = __get_authorization_headers()
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    downloaded_bytes: bytes = await response.read()
                    downloaded_bytes_len = len(downloaded_bytes)
                    return DownloadResult(uri=gsutil_uri,
                                          status=str(response.status),
                                          size=downloaded_bytes_len,
                                          downloaded_bytes=downloaded_bytes)
                else:
                    return DownloadResult(uri=gsutil_uri, status=str(response.status), size=0)
        except aiohttp.client_exceptions.ClientConnectorError:
            # https://stackoverflow.com/questions/57046073/why-do-i-get-the-error-clientconnectorerror
            return DownloadResult(uri=gsutil_uri, status=str(900), size=0)
        except asyncio.exceptions.TimeoutError:
            return DownloadResult(uri=gsutil_uri, status=str(901), size=0)
        except aiohttp.client_exceptions.ServerDisconnectedError:
            return DownloadResult(uri=gsutil_uri, status=str(902), size=0)
        except Exception:
            return DownloadResult(uri=gsutil_uri, status=str(999), size=0)


async def download_batch_to_dir(uri_batch: List[str],
                                destination_dir):
    async with aiohttp.ClientSession() as session:
        coroutines = [download_to_dir(gsutil_uri, destination_dir, session) for gsutil_uri in uri_batch]
        return await asyncio.gather(*coroutines)


async def download_batch_to_memory(uri_batch: List[str]):
    async with aiohttp.ClientSession() as session:
        coroutines = [download_to_memory(gsutil_uri, session) for gsutil_uri in uri_batch]
        return await asyncio.gather(*coroutines)


def download_batch_to_dir_wrapper(batch: DownloadBatch):
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(download_batch_to_dir(batch.gsutil_uri_batch, batch.destination_dir))
    result_dict = {}
    for result in results:
        result_dict[result.uri] = result

    return DownloadBatchResult(results=result_dict)


def download_batch_to_memory_wrapper(batch: DownloadBatch):
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(download_batch_to_memory(batch.gsutil_uri_batch))
    result_dict = {}
    for result in results:
        result_dict[result.uri] = result

    return DownloadBatchResult(results=result_dict)


class UploadResult:
    def __init__(self,
                 blob_name,
                 status,
                 uri):
        self.uri = uri
        self.blob_name = blob_name
        self.status = status


class UploadResource:
    def __init__(self,
                 bucket_name: str,
                 blob_name: str,
                 bytes_to_upload: bytes = None,
                 file_path_to_upload: str = None):
        self.bucket_name = bucket_name
        if bytes_to_upload and file_path_to_upload is None:
            raise Exception(f'Either bytes_to_upload or file_path_to_upload has to be provided.')

        self.blob_name = blob_name
        self.bytes_to_upload = bytes_to_upload
        self.file_path_to_upload = file_path_to_upload

    def get_uri(self):
        return make_uri(self.bucket_name, self.blob_name)


def upload_batch_wrapper(resources: List[UploadResource]) -> List[UploadResult]:
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(upload_batch(resources))
    return results


async def upload_batch(resources: List[UploadResource]):
    async with aiohttp.ClientSession() as session:
        coroutines = []
        for resource in resources:
            if resource.bytes_to_upload is not None:
                coroutines.append(upload_from_memory(resource.bytes_to_upload, resource.bucket_name, resource.blob_name, session))
            elif resource.file_path_to_upload is not None:
                coroutines.append(upload_from_file(resource.file_path_to_upload, resource.bucket_name, resource.blob_name, session))

        return await asyncio.gather(*coroutines)


async def upload_from_file(path_to_file: str,
                           bucket_name: str,
                           blob_name: str = None,
                           session: aiohttp.ClientSession = None) -> UploadResult:
    if not os.path.exists(path_to_file):
        raise Exception(f'{path_to_file=} file does not exist.')

    if blob_name is None:
        # will work only on unix-like file system
        blob_name = path_to_file.replace('/', '', 1)

    with open(path_to_file, 'rb') as file:
        file_bytes: bytes = file.read()
        return await upload_from_memory(file_bytes, bucket_name, blob_name, session)


async def upload_from_memory(bytes_to_upload: bytes,
                             bucket_name: str,
                             blob_name: str,
                             session: aiohttp.ClientSession = None) -> UploadResult:
    if session is None:
        async with aiohttp.ClientSession() as session:
            return await __upload_from_memory(bytes_to_upload, bucket_name, blob_name, session)
    else:
        return await __upload_from_memory(bytes_to_upload, bucket_name, blob_name, session)


async def __upload_from_memory(bytes_to_upload: bytes,
                               bucket_name: str,
                               blob_name: str,
                               session: aiohttp.ClientSession) -> UploadResult:
    headers = __get_authorization_headers()
    url = __make_cloud_api_put_url(bucket_name, blob_name)
    try:
        result = await session.put(url=url, headers=headers, data=bytes_to_upload)
        http_status = result.status
        if http_status == 200:
            gsutil_uri = make_uri(bucket_name, blob_name)
            return UploadResult(blob_name=blob_name, status=str(http_status), uri=gsutil_uri)
        return UploadResult(blob_name=blob_name, status=str(http_status), uri=None)
    except aiohttp.client_exceptions.ClientConnectorError:
        return UploadResult(blob_name=blob_name, status=str(900), uri=None)


def make_batches(strings: List[str], n=100):
    return [strings[i * n:(i + 1) * n] for i in range((len(strings) + n - 1) // n)]


def __get_authorization_headers() -> Dict[str, str]:
    access_token = __get_access_token()
    return {'Authorization': f'Bearer {access_token}'}


def __get_access_token() -> str:
    if CREDENTIALS.token is not None and CREDENTIALS.expired is False:
        return CREDENTIALS.token
    else:
        request = Request()
        CREDENTIALS.refresh(request)
        if CREDENTIALS.token is not None:
            return CREDENTIALS.token

        raise Exception('Failed to get OAuth2.0 Access Token to Cloud Storage')


def __make_cloud_api_get_url(uri: str):
    rest_api_url = uri.replace('gs://', CLOUD_API_URL)
    rest_api_url += '?alt=media'

    return rest_api_url


def __make_cloud_api_put_url(bucket_name: str, resource_name: str):
    # return f'{CLOUD_API_URL}storage/v1/b/{bucket_name}/o/{uri}'
    return f'{CLOUD_API_URL}{bucket_name}/{resource_name}?projection=full'


def validate_uri_list(uri_list: List[str]):
    errors = []
    result_uri_list = []
    for gsutil_uri in uri_list:
        try:
            parse_blob_uri(gsutil_uri)
        except InvalidGsutilUri as e:
            errors.append(e.args[0])

    if len(errors) > 0:
        raise InvalidGsutilUri(errors)

    return result_uri_list


def filter_out_existing_files(uri_list: List[str], destination_dir):
    existing = {}
    to_download = []

    for gsutil_uri in uri_list:
        bucket, resource_name = parse_blob_uri(gsutil_uri)
        local_file_path = get_local_file_path(destination_dir, bucket, resource_name)
        if exists(local_file_path):
            existing[gsutil_uri] = DownloadResult(uri=gsutil_uri,
                                                  file_name=local_file_path,
                                                  status="200",
                                                  size=getsize(local_file_path))
        else:
            to_download.append(gsutil_uri)

    return existing, to_download


def get_local_file_path(destination_dir, bucket, blob_name):
    return join(destination_dir, bucket, blob_name)


def make_uri(bucket_name, resource_name):
    return f'gs://{bucket_name}/{resource_name}'


def parse_blob_uri(uri: str):
    try:
        assert uri is not None and uri != ''
        bucket_and_resource_name = uri.replace('gs://', '').split('/', 1)
        assert len(bucket_and_resource_name) == 2

        bucket: str = bucket_and_resource_name[0]
        resource_name: str = bucket_and_resource_name[1]

        return bucket, resource_name
    except Exception:
        raise InvalidGsutilUri(f'{uri=} is not valid gsutil URI (File path to this resource in Cloud Storage)\n'
                               'Valid URI looks like this: gs://xplai-data-loadtests-europe-west4/reference/d/3/a/d3a1b201d30a8304.jpg')


class InvalidGsutilUri(Exception):
    pass


if __name__ != "__main__":
    # print(f'Info: new worker spawned pid={getpid()}')
    pass
