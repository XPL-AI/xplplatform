import os
import pytest
import uuid
from datetime import datetime
import pandas as pd
import random
import time

from xpl.data import config
from xpl.data.repositories import DataItemRepository, DataItem, RepositoryException, TableExistsException
from xpl.infrastructure.storage import Downloader


table_name_1d = 'autotest_experiment1d'
table_name_2d = 'digit_recognition'
table_name_3d = 'autotest_experiment3d'

dataset_id = 'xpl_auto_tests'


@pytest.mark.skip()
def test__get_audio():
    repository = DataItemRepository(dataset_id='xpl_test')
    data_frame = repository.select_as_data_frame(table_name='english_phoneme_recognition')
    files = list(data_frame['data_point_file'])
    Downloader.download(files, os.path.join(config.DATA_DIR, 'cache'), number_of_processes=4, coroutines_batch_size=20,
                        do_not_download_if_file_exists=True)
    pass


if __name__ == '__main__':
    test__get_audio()
    exit()


@pytest.mark.skip(reason="Should run once when tables do not exist. Rest of the tests are built on assumption that tables exist.")
def test__setup__run_if_tables_are_not_created():
    repository = DataItemRepository(dataset_id=dataset_id)
    repository.create_table(table_name=table_name_2d, modality_type='image', include_text=True, include_value=True)
    repository.create_table(table_name=table_name_1d, modality_type='audio')
    repository.create_table(table_name=table_name_3d, modality_type='3d')


def test__create_table__table_exists__raise_exception():
    repository = DataItemRepository(dataset_id=dataset_id)

    with pytest.raises(TableExistsException):
        repository.create_table(table_name=table_name_2d, modality_type='image')

    with pytest.raises(TableExistsException):
        repository.create_table(table_name=table_name_1d, modality_type='audio')

    with pytest.raises(TableExistsException):
        repository.create_table(table_name=table_name_3d, modality_type='3d')


def test__1d__insert__select_all__data_is_consistent():
    repository = DataItemRepository(dataset_id=dataset_id)

    """
        Create data items with common random predictor_id.
        predictor_id will be used to identify records that were created by this test from other records in BigQuery table. 
    """
    predictor_id = f'{random.random()}'
    data_items = [__new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:0000001', predictor_id=predictor_id)
                  ]

    """SUT: Insert DataItems to BigQuery"""
    table_name = table_name_1d
    repository.insert(data_items, table_name=table_name)

    """SUT: Select DataItems from BigQuery in DataItems object representation"""
    in_table_data_items = repository.select(table_name=table_name, predictor_id=predictor_id)

    """Verification"""
    """Expected: generated DataItem objects match those returned by select"""
    in_table_data_items_in_data_frame = __make_data_frame(in_table_data_items)
    data_items_in_data_frame = __make_data_frame(data_items)

    comparison_result = data_items_in_data_frame['data_point_id'].compare(in_table_data_items_in_data_frame['data_point_id'])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['data_point_file'].compare(in_table_data_items_in_data_frame['data_point_file'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['concept_id'].compare(in_table_data_items_in_data_frame['concept_id'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['instance_id'].compare(in_table_data_items_in_data_frame['instance_id'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['predictor_type'].compare(in_table_data_items_in_data_frame['predictor_type'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['predictor_id'].compare(in_table_data_items_in_data_frame['predictor_id'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['input_set'].compare(in_table_data_items_in_data_frame['input_set'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['log_informativeness'].compare(in_table_data_items_in_data_frame['log_informativeness'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['logit_confidence'].compare(in_table_data_items_in_data_frame['logit_confidence'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['previous_data_point_id'].compare(in_table_data_items_in_data_frame['previous_data_point_id'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['next_data_point_id'].compare(in_table_data_items_in_data_frame['next_data_point_id'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['collected_by_device_fingerprint_id'].compare(in_table_data_items_in_data_frame['collected_by_device_fingerprint_id'])])

    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['center_x'].compare(in_table_data_items_in_data_frame['center_x'])])
    comparison_result = pd.concat([comparison_result,
                                   data_items_in_data_frame['half_width'].compare(in_table_data_items_in_data_frame['half_width'])])

    assert comparison_result.empty

    """SUT: Select DataItems from BigQuery in pandas.DataFrame representation"""
    in_table_data_frame = repository.select_as_data_frame(table_name=table_name, predictor_id=predictor_id)
    comparison_result = in_table_data_frame['center_x'].compare(in_table_data_items_in_data_frame['center_x'])
    assert comparison_result.empty
    pass


def test__1d__insert_data_frame__select_all__data_is_consistent():
    pass


def test__2d__insert_data_frame__select_inserted__data_is_consistent():
    repository = DataItemRepository(dataset_id=dataset_id)

    """
        Create data items with common random predictor_id.
        predictor_id will be used to identify records that were created by this test from other records in BigQuery table. 
    """
    predictor_id = f'{random.random()}'
    data_items = [__new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:3216548', predictor_id=predictor_id)
                  ]

    """SUT: Insert DataItems from DataFrame to BigQuery"""
    # df = pd.read_csv(dataset2d_csv_path)
    table_name = table_name_2d
    data_items_in_data_frame = __make_data_frame(data_items)
    repository.insert_from_data_frame(data_items_in_data_frame, table_name=table_name)

    """SUT: Select DataItems from BigQuery in DataItems object representation"""
    in_table_data_items = repository.select(table_name=table_name, predictor_id=predictor_id)

    """Verification"""
    """Expected: generated DataItem objects match those returned by select"""
    in_table_data_items_in_data_frame = __make_data_frame(in_table_data_items)
    comparison_result = data_items_in_data_frame['center_y'].compare(in_table_data_items_in_data_frame['center_y'])
    assert comparison_result.empty
    comparison_result = data_items_in_data_frame['half_height'].compare(in_table_data_items_in_data_frame['half_height'])
    assert comparison_result.empty

    """SUT: Select DataItems from BigQuery in pandas.DataFrame representation"""
    in_table_data_frame = repository.select_as_data_frame(table_name=table_name, predictor_id=predictor_id)
    comparison_result = in_table_data_frame['center_x'].compare(in_table_data_items_in_data_frame['center_x'])
    assert comparison_result.empty
    comparison_result = in_table_data_frame['half_height'].compare(in_table_data_items_in_data_frame['half_height'])
    assert comparison_result.empty


def test__2d__insert__select_inserted__data_is_consistent():
    repository = DataItemRepository(dataset_id=dataset_id)

    """
        Create data items with common random predictor_id.
        predictor_id will be used to identify records that were created by this test from other records in BigQuery table. 
    """
    predictor_id = f'{random.random()}'
    data_items = [__new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id)
                  ]
    data_items_in_data_frame = __make_data_frame(data_items)

    """SUT: Insert DataItems to BigQuery"""
    table_name = table_name_2d
    repository.insert(data_items, table_name=table_name)

    """SUT: Select DataItems from BigQuery in DataItems object representation"""
    in_table_data_items = repository.select(table_name=table_name, predictor_id=predictor_id)

    """Verification"""
    """Expected: generated DataItem objects match those returned by select"""
    in_table_data_items_in_data_frame = __make_data_frame(in_table_data_items)
    comparison_result = data_items_in_data_frame['center_x'].compare(in_table_data_items_in_data_frame['center_x'])
    assert comparison_result.empty

    """SUT: Select DataItems from BigQuery in pandas.DataFrame representation"""
    in_table_data_frame = repository.select_as_data_frame(table_name=table_name, predictor_id=predictor_id)
    comparison_result = in_table_data_frame['center_x'].compare(in_table_data_items_in_data_frame['center_x'])
    assert comparison_result.empty


def test__select_by_row_id____data_is_consistent():
    repository = DataItemRepository(dataset_id=dataset_id)

    """
        Create data items with common random predictor_id.
        predictor_id will be used to identify records that were created by this test from other records in BigQuery table. 
    """
    predictor_id = f'{random.random()}'
    data_items = [__new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id),
                  __new_2d_data_item(concept_id='bn:0000002', predictor_id=predictor_id)
                  ]
    data_items_in_data_frame = __make_data_frame(data_items)

    """SUT: Insert DataItems to BigQuery"""
    table_name = table_name_2d
    repository.insert(data_items, table_name=table_name)

    """SUT: Select DataItems from BigQuery in DataItems object representation"""
    in_table_data_items = repository.select(table_name=table_name, predictor_id=predictor_id)

    itr = iter(in_table_data_items)
    first = next(itr)
    second = next(itr)

    in_table_data_items_by_row_ids = repository.select(table_name=table_name, row_ids=[first.row_id, second.row_id])
    assert len(in_table_data_items_by_row_ids) == 2
    assert in_table_data_items_by_row_ids[0].row_id == first.row_id
    assert in_table_data_items_by_row_ids[1].row_id == second.row_id


# TODO: Get back to those tests when there is decision on how to implement consistency.
#       On the general assumption those validations should happen at application logic level.
#       Can also be mitigated with @dataclasses
@pytest.mark.skip(reason="Isn't properly implemented and should move to application logic")
def test__2d__insert_3d_modality__exception():
    repository = DataItemRepository(dataset_id=dataset_id)
    predictor_id = f'{random.random()}'
    """ Create 3d DataItems """
    data_items = [__new_3d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_3d_data_item(concept_id='bn:3216548', predictor_id=predictor_id)
                  ]
    with pytest.raises(RepositoryException):
        """SUT: Insert DataItems to BigQuery"""
        repository.insert(data_items, table_name=table_name_2d)


@pytest.mark.skip(reason="Isn't properly implemented and should move to application logic")
def test__2d__insert_1d_modality__exception():
    repository = DataItemRepository(dataset_id='autotest_dataset')

    """
        Create 3d DataItems
    """
    predictor_id = f'{random.random()}'
    data_items = [__new_1d_data_item(concept_id='bn:3216548', predictor_id=predictor_id),
                  __new_1d_data_item(concept_id='bn:3216548', predictor_id=predictor_id)
                  ]

    """SUT: Insert DataItems to BigQuery"""
    repository.insert(data_items, table_name=table_name_2d)


def __make_data_frame(data_items: [DataItem]):
    items = []
    for i in data_items:
        item = i.dict()
        item['row_id'] = str(uuid.uuid4())
        if 'location' in item:
            location = item['location']
            if location is not None:
                item = {**item, **location}
                del item['location']
        items.append(item)

    return pd.DataFrame(items)


def __new_data_item(concept_id, predictor_id):
    """
        A factory method to create a DataItem.
        concept_id and predictor_id are set by particular test.
        rest of parameters are assigned random values.
    """
    data_point_id = str(uuid.uuid4())
    data_point_file = f'gs://xplai-samples-dev-europe-north1-01/' \
                      f'{data_point_id[0]}/{data_point_id[0]}/{data_point_id[0]}/' \
                      f'{data_point_id}'
    data_item = DataItem(data_point_id=data_point_id,
                         data_point_file=data_point_file,
                         timestamp=datetime.utcnow(),
                         concept_id=concept_id,
                         instance_id=str(uuid.uuid4()),
                         predictor_type='dataset',
                         predictor_id=predictor_id,
                         input_set='train',
                         log_informativeness=0,
                         logit_confidence=1,
                         previous_data_point_id=str(uuid.uuid4()),
                         next_data_point_id=str(uuid.uuid4()),
                         collected_by_device_fingerprint_id=str(uuid.uuid4()))

    time.sleep(0.025)
    return data_item


def __new_1d_data_item(concept_id, predictor_id):
    """
        A factory method to create a DataItem.
        concept_id and predictor_id are set by particular test.
        rest of parameters are assigned random values.
    """
    data_item = __new_data_item(concept_id, predictor_id)
    data_item.location = {
        'center_x': random.random(),
        'half_width': random.random()
    }
    time.sleep(0.025)
    return data_item


def __new_2d_data_item(concept_id, predictor_id):
    """
        A factory method to create a DataItem.
        concept_id and predictor_id are set by particular test.
        rest of parameters are assigned random values.
    """
    data_item = __new_data_item(concept_id, predictor_id)
    data_item.text = 'text'
    data_item.value = 0.5
    data_item.location = {
        'center_x': random.random(),
        'center_y': random.random(),
        'half_width': random.random(),
        'half_height': random.random(),
    }
    return data_item


def __new_3d_data_item(concept_id, predictor_id):
    """
            A factory method to create 3d DataItem.
            concept_id and predictor_id are set by particular test.
            rest of parameters are assigned random values.
    """
    data_item = __new_data_item(concept_id, predictor_id)
    data_item.location = {
        'center_x': random.random(),
        'center_y': random.random(),
        'center_t': random.random(),
        'half_width': random.random(),
        'half_height': random.random(),
        'half_period': random.random(),
    }
    return data_item


_RFC3339_MICROS = "%Y-%m-%dT%H:%M:%S.%f"
dataset2d_csv_path = os.path.join(os.path.dirname(__file__), 'dataset2d.csv')


def __timestamp_to_json_row(value):
    """Coerce 'value' to an JSON-compatible representation."""
    value = datetime.strptime(value, "%Y-%m-%d_%H-%M-%S")
    if isinstance(value, datetime):
        # For naive datetime objects UTC timezone is assumed, thus we format
        # those to string directly without conversion.
        # if value.tzinfo is not None:
        #     value = value.astimezone(utc)
        value = value.strftime(_RFC3339_MICROS)
    return value


def __playground_method():
    """a few useful snippets. TODO: will be deleted"""
    df = pd.read_csv(dataset2d_csv_path)
    # pandas
    # HOWTO rename columns in dataframe
    df.rename(columns={'data_point_uuid': 'data_point_id',
                       'utc_timestamp': 'timestamp',
                       'instance_uuid': 'instance_id',
                       'previous_data_point_uuid': 'previous_data_point_id'},
              inplace=True)

    # set value to all cells in a column
    df['predictor_type'] = 'dataset'

    # convert timestamp to BiqQuery friendly
    # apply function to all cells in the column
    df['timestamp'] = df['timestamp'].apply(lambda x: __timestamp_to_json_row(x))

    # squash columns in one column containing dictionary of the values of each row of original columns.
    location_columns = ['center_x', 'center_y', 'half_width', 'half_height']
    df['location'] = df[location_columns].dropna().apply(lambda row: dict(row), axis=1)
    df.drop(location_columns, inplace=True, axis=1)
