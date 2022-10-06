import random
from builtins import Exception
from datetime import datetime

from os import listdir, makedirs
from os.path import join, isfile, isdir, exists
import pandas
import psutil
import traceback
import time
from typing import List

import requests
import shutil
import sys

from google.cloud.storage.bucket import Bucket

from xpl.infrastructure.storage import config, UNIT
from xpl.infrastructure.storage.repository import CloudStorageRepository, Downloader, Uploader
from xpl.infrastructure.storage.repository_async import Downloader as DownloaderAsync, SingleProcessDownloader

LOAD_TEST_DATA_DIR = join(config.DATA_DIR, UNIT)
makedirs(LOAD_TEST_DATA_DIR, exist_ok=True)


def test_upload(caller_location, cpu_model, number_of_cores, bucket_name):
    files = prepare_files(subdirectory='EUROPE-WEST4c__EUROPE-WEST4/32_processes/reference')

    repository = CloudStorageRepository(bucket_name=bucket_name)
    bucket: Bucket = repository.get_bucket_details()
    bucket_location = bucket.location
    bucket_storage_class = bucket.storage_class
    processes = [32, 16, 8, 4, 2]

    total_transferred_files = 0
    total_lost_files = 0
    total_transferred_bytes = 0
    total_lost_bytes = 0
    total_processed = 0

    start = time.perf_counter()
    for number_of_processes in processes:
        storage_directory = f'upload_{caller_location}__{number_of_processes}'
        try:
            with Uploader(bucket=bucket_name, directory=storage_directory, files_to_upload=files, number_of_processes=number_of_processes) \
                    as uploader:
                for uploaded_item_result in uploader:
                    total_processed += 1
                    if uploaded_item_result.result == 'OK':
                        total_transferred_files += 1
                        total_transferred_bytes += uploaded_item_result.size
                    else:
                        total_lost_files += 1
                        total_lost_bytes += uploaded_item_result.size
                    do_print = total_transferred_files % 100 == 0
                    if do_print:
                        __print_progress(total_transferred_files, total_transferred_bytes, total_lost_files, total_lost_bytes)

        except Exception as e:
            error_msg = '\n'
            error_msg += str(e)
            error_msg += '\n\n'
            error_msg += traceback.format_exc()
            log_error(error_msg)
            raise e
        finally:
            log_results_to_csv(operation='upload',
                               caller_location=caller_location,
                               bucket_location=bucket_location,
                               storage_class=bucket_storage_class,
                               cpu_model=cpu_model,
                               number_of_cores=number_of_cores,
                               number_of_processes=number_of_processes,
                               total_transferred_files=total_transferred_files,
                               total_transferred_bytes=total_transferred_bytes,
                               total_lost_files=total_lost_files,
                               total_lost_bytes=total_lost_bytes,
                               time_in_seconds=int(time.perf_counter() - start))


def download_multiprocessing_test(caller_location, cpu_model, bucket_name, number_of_items_to_download, processes=None):
    if processes is None:
        processes = [32, 16, 8, 4, 2]

    cloud_uris = __get_uri_batch_from_file_in_storage(bucket_name=bucket_name, take=number_of_items_to_download)

    repository = CloudStorageRepository(bucket_name=bucket_name)
    bucket: Bucket = repository.get_bucket_details()
    bucket_location = bucket.location
    bucket_storage_class = bucket.storage_class

    number_of_cores = psutil.cpu_count(logical=False)
    number_of_virtual_cores = psutil.cpu_count()
    ram_total = psutil.virtual_memory().total

    psutil.cpu_percent()

    for number_of_processes in processes:
        print(f'download_multiprocessing_test {number_of_processes=}\n')

        local_directory = join(LOAD_TEST_DATA_DIR, f'{caller_location}__{bucket_location}')
        __erase_directory(local_directory)

        total_transferred_files = 0
        total_lost_files = 0
        total_transferred_bytes = 0
        total_lost_bytes = -1
        total_processed = 0

        cpu_percent_used_avg = psutil.cpu_percent()
        ram_used_avg = psutil.virtual_memory().used

        start = time.perf_counter()
        time_of_experiment = str(datetime.utcnow())
        try:
            with Downloader(cloud_file_uris=cloud_uris, local_directory=local_directory,
                            number_of_processes=number_of_processes) \
                    as downloader:
                for download_item_result in downloader:
                    total_processed += 1
                    if download_item_result.result == 'OK':
                        total_transferred_files += 1
                        total_transferred_bytes += download_item_result.size
                    else:
                        total_lost_files += 1
                        if download_item_result.error is not None:
                            log_error(download_item_result.error)
                        # total_lost_bytes += download_item_result.size
                    do_print = total_processed % 100 == 0

                    if do_print:
                        cpu_percent_used_avg = (cpu_percent_used_avg + psutil.cpu_percent()) / 2
                        ram_used_avg = (ram_used_avg + psutil.virtual_memory().used) / 2
                        __print_progress(total_transferred_files, total_transferred_bytes, total_lost_files, total_lost_bytes,
                                         int(time.perf_counter() - start), cpu_percent_used_avg, ram_used_avg)

                __print_progress(total_transferred_files, total_transferred_bytes, total_lost_files, total_lost_bytes,
                                 int(time.perf_counter() - start), cpu_percent_used_avg, ram_used_avg)
        except Exception as e:
            error_msg = '\n'
            error_msg += str(e)
            error_msg += '\n\n'
            error_msg += traceback.format_exc()
            log_error(error_msg)
            raise e
        finally:
            log_results_to_csv(operation='download_multiprocessing_test',
                               write_to='disk',
                               time_of_experiment=time_of_experiment,
                               caller_location=caller_location,
                               bucket_location=bucket_location,
                               storage_class=bucket_storage_class,
                               cpu_model=cpu_model,
                               number_of_cores=number_of_cores,
                               number_of_virtual_cores=number_of_virtual_cores,
                               total_ram=ram_total,
                               number_of_processes=number_of_processes,
                               coroutines_batch_size=0,
                               total_transferred_files=total_transferred_files,
                               total_transferred_bytes=total_transferred_bytes,
                               total_lost_files=total_lost_files,
                               total_lost_bytes=total_lost_bytes,
                               time_in_seconds=time.perf_counter() - start,
                               cpu_percent_used_avg=cpu_percent_used_avg,
                               ram_used_avg=ram_used_avg)


def download_multiprocessing_async_test(caller_location, cpu_model, bucket_name, number_of_items_to_download, coroutines_batch_sizes,
                                        number_of_processes):
    number_of_cores = psutil.cpu_count(logical=False)
    number_of_virtual_cores = psutil.cpu_count()
    ram_total = psutil.virtual_memory().total

    psutil.cpu_percent()

    repository = CloudStorageRepository(bucket_name=bucket_name)
    bucket: Bucket = repository.get_bucket_details()
    bucket_location = bucket.location
    bucket_storage_class = bucket.storage_class

    for batch_size in coroutines_batch_sizes:
        print_label = f'download_multiprocessing_async_test p:{number_of_processes} b:{batch_size}'
        local_directory = join(LOAD_TEST_DATA_DIR, f'{caller_location}__{bucket_location}')

        __erase_directory(local_directory)

        cloud_uris = __get_uri_batch_from_file_in_storage(bucket_name=bucket_name, take=number_of_items_to_download)

        cpu_percent_used_avg = psutil.cpu_percent()
        ram_used_avg = psutil.virtual_memory().used

        time_of_experiment = str(datetime.utcnow())
        total_to_download = len(cloud_uris)
        ok = {}
        ok_bytes = 0
        failed = {}
        total_processed = 0
        start = time.perf_counter()
        try:
            with DownloaderAsync(uri_list=cloud_uris, destination_dir=local_directory,
                                 number_of_processes=number_of_processes, coroutines_batch_size=batch_size) \
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
                    if batch_count % 5 == 0:
                        elapsed_time = time.perf_counter() - start
                        total_remaining = total_to_download - total_processed
                        estimated_time_left = total_remaining * elapsed_time / total_processed
                        cpu_percent_used_avg = (cpu_percent_used_avg + psutil.cpu_percent()) / 2
                        ram_used_avg = (ram_used_avg + psutil.virtual_memory().used) / 2
                        __print_progress2('Downloading..', len(ok), ok_bytes, len(failed), elapsed_time, estimated_time_left,
                                          cpu_percent_used_avg, ram_used_avg,
                                          download_label=print_label)
                    batch_count += 1
        except Exception as e:
            error_msg = '\n'
            error_msg += str(e)
            error_msg += '\n\n'
            error_msg += traceback.format_exc()
            log_error(error_msg)
            raise e
        finally:
            __print_progress2('Download complete', len(ok), ok_bytes, len(failed), elapsed_time, estimated_time_left,
                              cpu_percent_used_avg, ram_used_avg,
                              download_label=print_label, print_in_place=False)

            log_results_to_csv(operation='download_multiprocessing_async_test',
                               write_to='disk',
                               time_of_experiment=time_of_experiment,
                               caller_location=caller_location,
                               bucket_location=bucket_location,
                               storage_class=bucket_storage_class,
                               cpu_model=cpu_model,
                               number_of_cores=number_of_cores,
                               number_of_virtual_cores=number_of_virtual_cores,
                               total_ram=ram_total,
                               number_of_processes=number_of_processes,
                               coroutines_batch_size=batch_size,
                               total_transferred_files=len(ok),
                               total_transferred_bytes=ok_bytes,
                               total_lost_files=len(failed),
                               total_lost_bytes=-1,
                               time_in_seconds=time.perf_counter() - start,
                               cpu_percent_used_avg=cpu_percent_used_avg,
                               ram_used_avg=ram_used_avg)


def download_async_test(caller_location, cpu_model, bucket_name, number_of_items_to_download, coroutines_batch_sizes):
    number_of_cores = psutil.cpu_count(logical=False)
    number_of_virtual_cores = psutil.cpu_count()
    ram_total = psutil.virtual_memory().total

    psutil.cpu_percent()

    repository = CloudStorageRepository(bucket_name=bucket_name)
    bucket: Bucket = repository.get_bucket_details()
    bucket_location = bucket.location
    bucket_storage_class = bucket.storage_class

    for batch_size in coroutines_batch_sizes:
        print(f'download_async_test coroutines_batch_size={batch_size}\n')
        local_directory = join(LOAD_TEST_DATA_DIR, f'{caller_location}__{bucket_location}')

        __erase_directory(local_directory)

        cloud_uris = __get_uri_batch_from_file_in_storage(bucket_name=bucket_name, take=number_of_items_to_download)

        total_transferred_bytes = 0
        total_lost_bytes = -1
        total_processed = 0

        total_count_ok = 0
        total_count_fail = 0

        cpu_percent_used_avg = psutil.cpu_percent()
        ram_used_avg = psutil.virtual_memory().used

        time_of_experiment = str(datetime.utcnow())
        start = time.perf_counter()
        try:
            downloader = SingleProcessDownloader(uri_list=cloud_uris, destination_dir=local_directory,
                                                 coroutines_batch_size=batch_size)
            for download_item_result in downloader:
                total_processed += len(download_item_result.results)
                count_ok = 0
                bytes_ok = 0
                for result in download_item_result.results.values():
                    if result.status == '200':
                        count_ok += 1
                        bytes_ok += result.size
                        # bytes_ok += len(result.downloaded_bytes)
                # count_ok = list(download_item_result.results.values()).count('200')
                total_count_ok += count_ok
                total_transferred_bytes += bytes_ok
                total_count_fail += len(download_item_result.results) - count_ok
                if total_processed % 100 == 0:
                    cpu_percent_used_avg = (cpu_percent_used_avg + psutil.cpu_percent()) / 2
                    ram_used_avg = (ram_used_avg + psutil.virtual_memory().used) / 2
                    __print_progress(total_count_ok, total_transferred_bytes, total_count_fail, total_lost_bytes,
                                     int(time.perf_counter() - start), cpu_percent_used_avg, ram_used_avg)
        except Exception as e:
            error_msg = '\n'
            error_msg += str(e)
            error_msg += '\n\n'
            error_msg += traceback.format_exc()
            log_error(error_msg)
            raise e
        finally:
            __print_progress(total_count_ok, total_transferred_bytes, total_count_fail, total_lost_bytes,
                             int(time.perf_counter() - start), cpu_percent_used_avg, ram_used_avg)

            log_results_to_csv(operation='download_async_test',
                               write_to='disk',
                               time_of_experiment=time_of_experiment,
                               caller_location=caller_location,
                               bucket_location=bucket_location,
                               storage_class=bucket_storage_class,
                               cpu_model=cpu_model,
                               number_of_cores=number_of_cores,
                               number_of_virtual_cores=number_of_virtual_cores,
                               total_ram=ram_total,
                               number_of_processes=0,
                               coroutines_batch_size=batch_size,
                               total_transferred_files=total_count_ok,
                               total_transferred_bytes=total_transferred_bytes,
                               total_lost_files=total_count_fail,
                               total_lost_bytes=total_lost_bytes,
                               time_in_seconds=time.perf_counter() - start,
                               cpu_percent_used_avg=cpu_percent_used_avg,
                               ram_used_avg=ram_used_avg)


def log_results_to_csv(operation: str,
                       write_to: str,
                       time_of_experiment: str,
                       caller_location: str,
                       bucket_location: str,
                       storage_class: str,
                       cpu_model: str,
                       number_of_cores: int,
                       number_of_virtual_cores: int,
                       total_ram: int,
                       number_of_processes: int,
                       coroutines_batch_size: int,
                       total_transferred_files: int,
                       total_transferred_bytes: int,
                       total_lost_files: int,
                       total_lost_bytes: int,
                       time_in_seconds: float,
                       cpu_percent_used_avg: float,
                       ram_used_avg: float
                       ):
    log_file_name = join(LOAD_TEST_DATA_DIR, "load_test__results.csv")

    log_df: pandas.DataFrame
    if isfile(log_file_name):
        log_df = pandas.read_csv(log_file_name)
    else:
        log_df = pandas.DataFrame(columns=['operation',
                                           'write_to',
                                           'time_of_experiment',
                                           'caller_location',
                                           'bucket_location',
                                           'storage_class',
                                           'cpu_model',
                                           'number_of_cores',
                                           'number_of_virtual_cores',
                                           'total_ram',
                                           'number_of_processes',
                                           'coroutines_batch_size',
                                           'total_transferred_files',
                                           'total_transferred_bytes',
                                           'total_lost_files',
                                           'total_lost_bytes',
                                           'time_in_seconds',
                                           'cpu_percent_used_avg',
                                           'ram_used_avg'
                                           ])

    row = {
        'operation': operation,
        'write_to': write_to,
        'time_of_experiment': time_of_experiment,
        'caller_location': caller_location,
        'bucket_location': bucket_location,
        'storage_class': storage_class,
        'cpu_model': cpu_model,
        'number_of_cores': number_of_cores,
        'number_of_virtual_cores': number_of_virtual_cores,
        'total_ram': total_ram,
        'number_of_processes': number_of_processes,
        'coroutines_batch_size': coroutines_batch_size,
        'total_transferred_files': total_transferred_files,
        'total_transferred_bytes': total_transferred_bytes,
        'total_lost_files': total_lost_files,
        'total_lost_bytes': total_lost_bytes,
        'time_in_seconds': time_in_seconds,
        'cpu_percent_used_avg': cpu_percent_used_avg,
        'ram_used_avg': ram_used_avg
    }

    log_df = log_df.append(row, ignore_index=True)
    log_df.to_csv(log_file_name, index=False)


def log_error(error):
    with open(join(LOAD_TEST_DATA_DIR, "download_upload__load_test__log.txt"), "a") as log_file:
        log_entry = str(error)
        log_file.write(log_entry)


def prepare_files(subdirectory) -> List[str]:
    """
    scans for files in the user experiment directory
    Only used for this load test scenario
    """
    local_dir = join(LOAD_TEST_DATA_DIR, subdirectory)
    return get_list_of_files(local_dir)


def prepare_files_to_download(data_subdir, user_id, experiment_id, bucket, remote_dir) -> List[str]:
    """
    scans for files in the user experiment directory
    Only used for this load test scenario
    """
    root_data_dir = join(config.DATA_DIR, data_subdir)
    user_id = user_id
    experiment_id = experiment_id
    local_dir = join(root_data_dir, user_id, experiment_id)
    files = [str(join('gs://', bucket, remote_dir, str(f[0]), str(f[1]), str(f[2]), str(f)))
             for f in listdir(local_dir) if isfile(join(local_dir, f))]

    return files


def __print_progress(total_transferred_files, total_transferred_bytes, total_lost_files, total_lost_bytes, seconds=0,
                     cpu_percent_used_avg=0, ram_used_avg=0):
    sys.stdout.write("files_OK: [%d]  bytes_OK: [%d]  files_FAIL: [%d]  bytes_FAIL: [%d]  "
                     "time passed(sec): [%d]  avg_cpu [%d] avg_ram [%d]MB \r" %
                     (total_transferred_files, total_transferred_bytes, total_lost_files, total_lost_bytes,
                      seconds, cpu_percent_used_avg, int(ram_used_avg / 1024 / 1024)))
    sys.stdout.flush()


def __print_progress2(status, ok, ok_bytes, failed, elapsed, remaining, cpu_percent_used_avg, ram_used_avg,
                      print_in_place=True, already_existed=0,
                      download_label: str = ''):
    speed = ok_bytes / elapsed
    speed_in_files = ok / elapsed
    if print_in_place:
        sys.stdout.write("%s: [%s]  "
                         "OK: [%d]  MB OK: [%d]  FAILED: [%d]  Elapsed: [%d]s, Remaining: [%d]s  AVG Speed: [%f]MB/s [%d]files/s "
                         "avg_cpu [%d] avg_ram [%d]MB\r" %
                         (download_label, status, ok, ok_bytes / 1024 / 1024, failed, elapsed, remaining, speed / 1024 / 1024,
                          speed_in_files, cpu_percent_used_avg, ram_used_avg))
        sys.stdout.flush()
    else:
        print("%s: [%s]  OK: [%d]  MB OK: [%d]  "
              "FAILED: [%d]  EXISTED: [%d]  Elapsed: [%d]s, Remaining: [%d]s  AVG Speed: [%f]MB/s [%d]files/s "
              "avg_cpu [%d] avg_ram [%d]MB \r" %
              (download_label, status, ok, ok_bytes / 1024 / 1024, failed, already_existed, elapsed, remaining, speed / 1024 / 1024,
               speed_in_files, cpu_percent_used_avg, ram_used_avg))


def __read_uri_file(uri_file) -> List[str]:
    uris: List[str]
    with open(uri_file) as f:
        uris = f.readlines()
    uris = [x.strip() for x in uris]
    return uris


def __get_uri_batch_from_file_in_storage(bucket_name, take: int = None, randomize: bool = False) -> List[str]:
    repository = CloudStorageRepository(bucket_name=bucket_name)
    try:
        text = repository.download_as_text(blob_name='reference-files.txt')
    except requests.exceptions.ReadTimeout:
        text = repository.download_as_text(blob_name='reference-files.txt')

    all_uris = text.splitlines(keepends=False)

    if take is None:
        return all_uris

    if randomize is True:
        return random.sample(all_uris, take)

    return all_uris[:take]


def __erase_directory(directory):
    start = time.perf_counter()
    if exists(directory):
        shutil.rmtree(directory, ignore_errors=True)
    print(f'Directory cleaned up in {time.perf_counter() - start}  seconds\n')


def get_list_of_files(directory):
    """
    basically a copy-paste to get all files from subdirectories recursively.
    https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    """
    list_of_file = listdir(directory)
    all_files = list()
    for entry in list_of_file:
        full_path = join(directory, entry)
        # If entry is a directory then get the list of files in this directory
        if isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            if not full_path.endswith('.DS_Store'):
                all_files.append(full_path)

    return all_files


def run_test_set(test_set_name: str, caller_location: str, cpu_model: str, bucket_name: str, number_of_items_to_download: int):
    if test_set_name == 'download_multiprocessing_test':
        download_multiprocessing_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
                                      number_of_items_to_download=number_of_items_to_download,
                                      # processes=[16, 8, 4, 2, 1])
                                      processes=[16])

    if test_set_name == 'download_async_test':
        download_async_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
                            number_of_items_to_download=number_of_items_to_download,
                            # coroutines_batch_sizes=[1600, 800, 400, 200, 100, 50])
                            coroutines_batch_sizes=[1600])

    if test_set_name == 'download_multiprocessing_with_async_test':
        download_multiprocessing_async_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
                                            number_of_items_to_download=number_of_items_to_download,
                                            coroutines_batch_sizes=[800, 400, 200, 100, 50], number_of_processes=1)
        # download_multiprocessing_async_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
        #                                     number_of_items_to_download=number_of_items,
        #                                     coroutines_batch_sizes=[400, 200, 100, 50, 25], number_of_processes=2)
        # download_multiprocessing_async_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
        #                                     number_of_items_to_download=number_of_items,
        #                                     coroutines_batch_sizes=[200, 100, 50, 25, 12], number_of_processes=4)
        # download_multiprocessing_async_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
        #                                     number_of_items_to_download=number_of_items,
        #                                     coroutines_batch_sizes=[100, 50, 25, 12, 6], number_of_processes=8)
        # download_multiprocessing_async_test(caller_location=caller_location, cpu_model=cpu_model, bucket_name=bucket_name,
        #                                     number_of_items_to_download=number_of_items,
        #                                     coroutines_batch_sizes=[50, 25, 12, 6, 3], number_of_processes=16)


if __name__ == "__main__":
    location = 'Lidingo'
    cpu = '2 GHz Quad-Core Intel Core i5'
    storage_bucket = 'xplai-data-loadtests-europe-west4'
    number_of_items = 10000

    run_test_set(test_set_name='download_multiprocessing_test', caller_location=location, cpu_model=cpu,
                 bucket_name=storage_bucket, number_of_items_to_download=number_of_items)

    run_test_set(test_set_name='download_multiprocessing_with_async_test',
                 caller_location=location, cpu_model=cpu, bucket_name=storage_bucket, number_of_items_to_download=number_of_items)

    run_test_set(test_set_name='download_async_test', caller_location=location, cpu_model=cpu,
                 bucket_name=storage_bucket, number_of_items_to_download=number_of_items)
